def acumulate_lists(l1, l2, acum_step):
    acum_l1 = []
    acum_l2 = []
    current_l1 = []
    current_l2 = []
    for i in range(len(l2)):
        current_l1.append(l1[i])
        current_l2.append(l2[i])
        if (i + 1) % acum_step == 0 and i != 0:
            acum_l1.append(current_l1)
            acum_l2.append(current_l2)
            current_l1 = []
            current_l2 = []
    return acum_l1, acum_l2