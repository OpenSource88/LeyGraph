
def LeyGraph_Shared_In(in_sp1, in_sp2):
    if in_sp1.dataset != in_sp2.dataset or in_sp1.sampler_setting != in_sp2.sampler_setting:
        sp1 = in_sp1
        sp2 = in_sp2
        exist_tiny = 0
    elif in_sp1.batch_size == in_sp2.batch_size:
        sp1 = in_sp1
        sp2 = in_sp2
        sp1.num = 1
        sp2.num = 1
        exist_tiny = 1
    else:
        tmp_a = in_sp1.batch_size
        a = tmp_a
        tmp_b = in_sp2.batch_size
        b = tmp_b
        while b != 0:
            a, b = b, a%b
        sp1 = in_sp1
        sp2 = in_sp2
        sp1.batch_size = a
        sp2.batch_size = a
        sp1.num = int(tmp_a/a) 
        sp2.num = int(tmp_b/a)
        exist_tiny = 1

    return sp1, sp2, exist_tiny 

