import torch


def pad_and_batch(species_coordinates):
    max_atoms = max([c.shape[1] for _, c in species_coordinates])
    species = []
    coordinates = []
    for s, c in species_coordinates:
        natoms = c.shape[1]
        if len(s.shape) == 1:
            s = s.unsqueeze(0)
        if natoms < max_atoms:
            padding = torch.full((s.shape[0], max_atoms - natoms), -1,
                                 dtype=torch.long, device=s.device)
            s = torch.cat([s, padding], dim=1)
            padding = torch.full((c.shape[0], max_atoms - natoms, 3), 0,
                                 dtype=c.dtype, device=c.device)
            c = torch.cat([c, padding], dim=1)
        s = s.expand(c.shape[0], max_atoms)
        species.append(s)
        coordinates.append(c)
    return torch.cat(species), torch.cat(coordinates)


def present_species(species):
    present_species = species.flatten().unique(sorted=True)
    if present_species[0].item() == -1:
        present_species = present_species[1:]
    return present_species


def strip_redundant_padding(species, coordinates):
    non_padding = (species >= 0).any(dim=0)
    species = species.masked_select(non_padding, dim=1)
    coordinates = coordinates.masked_select(non_padding, dim=1)
    return species, coordinates