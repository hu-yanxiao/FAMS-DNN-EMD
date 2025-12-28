from typing import Dict, List, Optional

import torch
from e3nn.util.jit import compile_mode

from ptagnn.tools.scatter import scatter_sum


@compile_mode("script")
class LAMMPS_MACE(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer("atomic_numbers", model.atomic_numbers)
        self.register_buffer("r_max", model.r_max)
        self.register_buffer("num_interactions", model.num_interactions)
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        local_or_ghost: torch.Tensor,
        compute_virials: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        num_graphs = data["ptr"].numel() - 1
        compute_displacement = False
        if compute_virials:
            compute_displacement = True
        out = self.model(
            data,
            training=False,
            compute_force=False,
            compute_virials=False,
            compute_stress=False,
            compute_displacement=compute_displacement,
        )
        node_energy = out["node_energy"]
        if node_energy is None:
            return {
                "total_energy_local": None,
                "node_energy": None,
                "forces": None,
                "virials": None,
            }
        positions = data["positions"]
        displacement = out["displacement"]
        vectors=out["vectors"]
        forces: Optional[torch.Tensor] = torch.zeros_like(positions)
        virials: Optional[torch.Tensor] = torch.zeros_like(data["cell"])
        atom_virial: Optional[torch.Tensor] = None
        edge_forces:Optional[torch.Tensor] = None
        # accumulate energies of local atoms
        node_energy_local = node_energy * local_or_ghost
        total_energy_local = scatter_sum(
            src=node_energy_local, index=data["batch"], dim=-1, dim_size=num_graphs
        )
        # compute partial forces and (possibly) partial virials
        grad_outputs: List[Optional[torch.Tensor]] = [
            torch.ones_like(total_energy_local)
        ]
        if compute_virials and displacement is not None:
            forces, edge_forces = torch.autograd.grad(
                outputs=[total_energy_local],
                inputs=[positions, vectors],
                grad_outputs=grad_outputs,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
            if forces is not None:
                forces = -1 * forces
            else:
                forces = torch.zeros_like(positions)
            if edge_forces is not None:
                edge_forces = -1 * edge_forces 

        else:
            forces = torch.autograd.grad(
                outputs=[total_energy_local],
                inputs=[positions],
                grad_outputs=grad_outputs,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )[0]
            if forces is not None:
                forces = -1 * forces
            else:
                forces = torch.zeros_like(positions)
        edge_index=data["edge_index"]
        edge_virial = torch.einsum("zi,zj->zij", edge_forces, vectors)
        atom_virial_sender = scatter_sum(
        src=edge_virial, index=edge_index[0], dim=0, dim_size=num_atoms
        )
        atom_virial_receiver = scatter_sum(
        src=edge_virial, index=edge_index[1], dim=0, dim_size=num_atoms
        )
        atom_virial = (atom_virial_sender + atom_virial_receiver) / 2
        atom_virial = (atom_virial + atom_virial.transpose(-1, -2)) / 2
        atom_virial*=-1
        virials = atom_virial.sum(dim=0)
        
        return {
            "total_energy_local": total_energy_local,
            "node_energy": node_energy,
            "forces": forces,
            "virials": virials,
            "atomic_vairals":atom_virial,
        }
