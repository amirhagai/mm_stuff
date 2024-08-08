def new_optimizer_step_option_1(optim_wrapper, **kwargs) -> None:
    """
    only added weights being update 
    """
    if optim_wrapper.clip_grad_kwargs:
        optim_wrapper._clip_grad()
    for param in optim_wrapper.optimizer.param_groups:
        if 'bbox_head' in param['params'][0].param_name and 'rtm_cls' not in param['params'][0].param_name:
            param['params'][0].grad.data.zero_()
        elif 'rtm_cls' in param['params'][0].param_name and 'weight' in param['params'][0].param_name:
            param['params'][0].grad.data[:15, :, :, :].zero_()
    optim_wrapper.optimizer.step(**kwargs)