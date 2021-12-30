from functools import partial


def args_print(p, bar=30):
    """
    argparseの parse_args() で生成されたオブジェクトを入力すると、
    integersとaccumulateを自動で取得して表示する
    [in] p: parse_args()で生成されたオブジェクト
    [in] bar: 区切りのハイフンの数
    """

    print('-' * bar)
    args = [(i, getattr(p, i)) for i in dir(p) if '_' not in i[0]]
    for i, j in args:
        if isinstance(j, list):
            print('{0}[{1}]:'.format(i, len(j)))
            [print('\t{}'.format(k)) for k in j]
        else:
            print('{0}:\t{1}'.format(i, j))

    print('-' * bar)


def reorder_from_idx(idx, input_list):
    """インデックスidxからの順番にし直す

    input [a, b, c, d, e] -> if idx = 2 then [c, d, e, a, b]
    """
    len_a = len(input_list)
    return [(i + idx) % len_a for i in input_list]


def cyclic_perm_index(input_list):
    """return 巡回インデックス
    """
    return [partial(reorder_from_idx, i)(input_list) for i in range(len(input_list))]


def swap_positions(input_list, pos1, pos2):
    """Swap elements indexed by pos1 and pos2
    """
    input_list[pos1], input_list[pos2] = input_list[pos2], input_list[pos1]
    return input_list


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
