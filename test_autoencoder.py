from autoencoder import ConvAutoEncoder, VAE, ResNet_VAE, MaskedAutoencoderViT
import torch
import torchinfo

def test_conv_autoencoder():
    model = ConvAutoEncoder(pretrained="vgg")
    assert model is not None
    assert model.encoder is not None
    assert model.decoder is not None
    # show model
    torchinfo.summary(model, input_size=(1, 3, 224, 224))
    # test forward pass
    # get image
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    print(out.shape)
    assert out is not None
    assert out.shape == (1, 3, 224, 224)

def test_vae():
    model = VAE(input_dim=224, latent_dim=100, hidden_dim=256)
    assert model is not None
    assert model.encoder is not None
    assert model.decoder is not None
    # show model
    torchinfo.summary(model.encoder, input_size=(1, 3, 224, 224))
    # test forward pass
    # get image
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    assert out is not None

def test_pretrained_vae():
    model = ResNet_VAE()
    assert model is not None
    # show model
    torchinfo.summary(model, input_size=(1, 3, 224, 224))
    # test forward pass
    # get image
    x = torch.randn(2, 3, 224, 224) # needs more batches due to batch normalization, maybe change that in future (reinforcement learning)
    out = model(x)
    assert out is not None

def test_masked_ae():
    model = MaskedAutoencoderViT()
    assert model is not None
    # show model
    torchinfo.summary(model, input_size=(1, 3, 224, 224))
    # test forward pass
    # get image
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    assert out is not None

#test_vae()
#test_pretrained_vae()
#test_masked_ae()
#test_conv_autoencoder()