import torch


def sample_episode(
    dataset,
    n_way: int = 2,
    k_shot: int = 5,
    n_query: int = 15,
    allow_replacement: bool = False,
):
    unique_labels = list(dataset.class_to_indices.keys())
    if len(unique_labels) < n_way:
        raise ValueError(f"类别数量不足: 需要 {n_way} 类, 当前 {len(unique_labels)} 类")

    selected = torch.randperm(len(unique_labels))[:n_way].tolist()
    selected_labels = [unique_labels[i] for i in selected]

    support_images, support_labels = [], []
    query_images, query_labels = [], []

    for episode_label, dataset_label in enumerate(selected_labels):
        indices = dataset.class_to_indices[dataset_label]
        required = k_shot + n_query
        if len(indices) < required and not allow_replacement:
            raise ValueError(
                f"类别 {dataset_label} 样本不足: 需要 {required}, 当前 {len(indices)}"
            )

        idx_tensor = torch.tensor(indices, dtype=torch.long)
        if len(indices) >= required:
            shuffled = idx_tensor[torch.randperm(len(indices))]
            support_indices = shuffled[:k_shot]
            query_indices = shuffled[k_shot : k_shot + n_query]
        else:
            sampled = idx_tensor[torch.randint(0, len(indices), (required,))]
            support_indices = sampled[:k_shot]
            query_indices = sampled[k_shot:]

        for idx in support_indices:
            img, _ = dataset[int(idx)]
            support_images.append(img)
            support_labels.append(episode_label)
        for idx in query_indices:
            img, _ = dataset[int(idx)]
            query_images.append(img)
            query_labels.append(episode_label)

    return (
        torch.stack(support_images, dim=0),
        torch.tensor(support_labels, dtype=torch.long),
        torch.stack(query_images, dim=0),
        torch.tensor(query_labels, dtype=torch.long),
    )
