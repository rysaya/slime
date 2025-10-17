import ray
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.wandb_utils import init_wandb_primary


def train(args):
    # allocate the GPUs
    pgs = create_placement_groups(args)
    wandb_run_id = init_wandb_primary(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    init_gen_engine = (
        args.train_type == "rl" or (args.eval_files is not None and args.eval_interval > 0)
    ) and not args.debug_train_only
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(
        args, pgs["rollout"], wandb_run_id=wandb_run_id, init_gen_engines=init_gen_engine
    )

    # create the actor and critic models
    actor_model, critic_model = create_training_models(args, pgs, wandb_run_id=wandb_run_id)

    if init_gen_engine and not args.debug_rollout_only:
        ray.get(actor_model.async_init_weight_update_connections(rollout_manager))

    actor_model.set_rollout_manager(rollout_manager)

    if args.colocate:
        ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_WEIGHTS]))

    # always update weight first so that sglang has the loaded weights from training.
    actor_model.update_weights()

    if args.offload:
        ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_KV_CACHE]))

    # If not colocate, use async train to save time
    if not args.colocate:
        rollout_data_next_future = rollout_manager.async_generate(args.start_rollout_id)
    # make eval at first step
    need_eval = args.eval_interval > 0
    need_on_off_switch = args.colocate
    if need_eval:
        ray.get(rollout_manager.async_eval(args.start_rollout_id))
    # note that for async training, one can change the position of the sync operation(ray.get).
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        if args.colocate:
            rollout_data_curr_ref = ray.get(rollout_manager.generate.remote(rollout_id))
            if need_on_off_switch:
                # TODO: face OOm issue when merging these two ray get. Split them and debug later
                ray.get(rollout_manager.async_offload())
                ray.get(actor_model.async_onload())
        else:
            rollout_data_curr_ref = ray.get(rollout_data_next_future)
            if rollout_id + 1 < args.num_rollout:
                rollout_data_next_future = rollout_manager.async_generate(rollout_id + 1)

        if args.use_critic:
            critic_train_handle = critic_model.async_train(rollout_id, rollout_data_curr_ref)
            if rollout_id >= args.num_critic_only_steps:
                ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref))
            ray.get(critic_train_handle)
        else:
            ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref))
        if args.save_interval is not None and (
            (rollout_id + 1) % args.save_interval == 0
            or (num_rollout_per_epoch is not None and (rollout_id + 1) % num_rollout_per_epoch == 0)
        ):
            ray.get(
                actor_model.async_save_model(rollout_id)
                + [rollout_manager.train_data_loader.save.remote(rollout_id)]
                + critic_model.save_model(rollout_id)
                if args.use_critic
                else []
            )

        need_eval = args.eval_interval > 0 and (
            (rollout_id + 1) % args.eval_interval == 0
            or (num_rollout_per_epoch is not None and (rollout_id + 1) % num_rollout_per_epoch == 0)
        )
        need_on_off_switch = args.colocate and (not args.turn_off_train_update_weights or need_eval)
        if args.colocate:
            if need_on_off_switch:
                ray.get(actor_model.async_offload())
                if args.use_critic:
                    ray.get(critic_model.async_offload())
                ray.get(rollout_manager.async_onload(tags=[GPU_MEMORY_TYPE_WEIGHTS]))
        else:
            rollout_data_next_future = ray.wait([rollout_data_next_future], num_returns=1)[0][0]

        if need_on_off_switch:
            ray.get(actor_model.async_update_weights())
            ray.get(rollout_manager.async_onload(tags=[GPU_MEMORY_TYPE_KV_CACHE]))

        if need_eval:
            ray.get(rollout_manager.async_eval(rollout_id))
    ray.get(rollout_manager.dispose.remote())


if __name__ == "__main__":
    args = parse_args()
    train(args)
