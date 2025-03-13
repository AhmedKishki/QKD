exp1
    retrials = 4
    kd_loss_labels = ['CS', 'KL']
    alpha_st_pairs = [(1.0,0.5)]
    temperatures = [6.0]
    max_lr = 1e-3
    min_lr = 1e-6
    teacher_lr = 1e-6
    num_epochs = [  (00, 100, 00) ]
    names = ["_".join(f"{x:02d}" for x in t) for t in num_epochs]

exp2
    retrials = 4
    kd_loss_labels = ['CS', 'KL']
    alpha_st_pairs = [(1.0,0.5)]
    temperatures = [6.0]
    max_lr = 1e-3
    min_lr = 1e-6
    teacher_lr = 1e-6
    num_epochs = [  (00, 0, 100) ]
    names = ["_".join(f"{x:02d}" for x in t) for t in num_epochs]