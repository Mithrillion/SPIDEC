import torch
import FrEIA.framework as Ff
import FrEIA.modules as Fm

def _get_sequential_inn(coupling_width, input_dims, n_inn_blocks):
    def subnet_fc(dims_in, dims_out):
        subnet = torch.nn.Sequential(
            torch.nn.Linear(dims_in, coupling_width),
            torch.nn.ReLU(),
            torch.nn.Linear(coupling_width, dims_out),
        )
        torch.nn.init.xavier_normal_(subnet[0].weight)
        torch.nn.init.constant_(subnet[-1].weight, 0.0)
        return subnet

    inn = Ff.SequenceINN(input_dims)
    for i in range(n_inn_blocks):
        inn.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, permute_soft=False)

    return inn