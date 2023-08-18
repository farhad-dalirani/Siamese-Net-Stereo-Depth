import torch 

def hinge_loss(pos_patch_similarity, neg_patch_similarity, margin=0.2):
    """Hinge Loss for training Siamese neural network for
       patch based stere matching 

    Args:
        pos_patch_similaritys: similarity of refrence patches with positive patches, tensor (batch, )
        neg_patch_similarity: similarity of refrence patches with negative patches, tensor (batch, )
        margin: it signifies the degree of confidence with which the model makes its predictions.
    """

    # hinge loss = max(0, neg_score + margin - pos_score).
    loss = torch.max(neg_patch_similarity + margin - pos_patch_similarity, torch.zeros_like(pos_patch_similarity))
    batch_mean_loss = loss.mean()

    # percentage of patches in th batch that similarity score to the positive patches are higer than negative patches.
    batch_accuracy = torch.sum(pos_patch_similarity > neg_patch_similarity) / len(pos_patch_similarity)

    return batch_mean_loss, batch_accuracy

if __name__ == '__main__':

    pos_patch_similarity = torch.rand(size=(10,))
    neg_patch_similarity = torch.rand(size=(10,))

    mean_loss, accuracy = hinge_loss(pos_patch_similarity, neg_patch_similarity, margin=0.2)

    print(pos_patch_similarity)
    print(neg_patch_similarity)
    print(mean_loss)
    print(accuracy)