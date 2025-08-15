import ray

from slime.ray.placement_group import create_actor_group, create_placement_groups, create_rollout_manager
from slime.utils.arguments import parse_args
from slime.utils.wandb_utils import init_wandb_primary
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS


def train(args):
    # allocate the GPUs
    pgs = create_placement_groups(args)
    wandb_run_id = init_wandb_primary(args)

    actor_model = create_actor_group(args, pgs["actor"], wandb_run_id=wandb_run_id)
    # sync the initialization (model initalization, load checkpoint, etc.)
    start_rollout_ids = ray.get(
        actor_model.async_init(args, role="actor", with_ref=args.kl_coef != 0 or args.use_kl_loss)
    )

    # create the rollout manager, with sglang engines inside.
    rollout_manager = create_rollout_manager(args, pgs["rollout"], actor_model, wandb_run_id=wandb_run_id)

    # calculate num_rollout from num_epoch
    num_rollout_per_epoch = None
    if args.num_rollout is None:
        num_rollout_per_epoch = ray.get(rollout_manager.controller.get_num_rollout_per_epoch.remote())
        args.num_rollout = num_rollout_per_epoch * args.num_epoch
    assert args.num_rollout > 0

    assert len(set(start_rollout_ids)) == 1
    if args.start_rollout_id is None:
        args.start_rollout_id = start_rollout_ids[0]

    if args.load is not None:
        ray.get(rollout_manager.controller.load.remote(args.start_rollout_id - 1))

    if args.colocate:
        ray.get(rollout_manager.async_onload(tags=[GPU_MEMORY_TYPE_WEIGHTS]))

    # always update weight first so that sglang has the loaded weights from training.
    ray.get(actor_model.async_update_weights())

    if args.colocate:
        ray.get(rollout_manager.async_onload(tags=[GPU_MEMORY_TYPE_KV_CACHE]))

    # If not colocate, use async train to save time
    if not args.colocate:
        rollout_data_next_future = rollout_manager.async_generate(args.start_rollout_id)
    # make eval at first step
    if args.eval_interval is not None:
        ray.get(rollout_manager.async_eval(args.start_rollout_id))
    # note that for async training, one can change the position of the sync operation(ray.get).
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        if args.colocate:
            rollout_data_curr_ref = ray.get(rollout_manager.async_generate(rollout_id))
            ray.get(rollout_manager.async_offload())
        else:
            rollout_data_curr_ref = ray.get(rollout_data_next_future)
            if rollout_id + 1 < args.num_rollout:
                rollout_data_next_future = rollout_manager.async_generate(rollout_id + 1)

        ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref))
        if args.save_interval is not None and (
            (rollout_id + 1) % args.save_interval == 0
            or (num_rollout_per_epoch is not None and (rollout_id + 1) % num_rollout_per_epoch == 0)
        ):
            ray.get(actor_model.async_save_model(rollout_id) + [rollout_manager.controller.save.remote(rollout_id)])
        if args.colocate:
            ray.get(actor_model.async_offload() + rollout_manager.async_onload(tags=[GPU_MEMORY_TYPE_WEIGHTS]))
        else:
            # sync generate before update weights to prevent update weight in the middle of generation
            if isinstance(rollout_data_next_future, list):
                rollout_data_next_future = ray.wait(
                    rollout_data_next_future, num_returns=len(rollout_data_next_future)
                )[0]
            else:
                rollout_data_next_future = ray.wait([rollout_data_next_future], num_returns=1)[0][0]
        ray.get(actor_model.async_update_weights())
        if args.colocate:
            ray.get(rollout_manager.async_onload(tags=[GPU_MEMORY_TYPE_KV_CACHE]))

        if args.eval_interval is not None and (
            (rollout_id + 1) % args.eval_interval == 0
            or (num_rollout_per_epoch is not None and (rollout_id + 1) % num_rollout_per_epoch == 0)
        ):
            ray.get(rollout_manager.async_eval(rollout_id))


if __name__ == "__main__":
    args = parse_args()
    train(args)
