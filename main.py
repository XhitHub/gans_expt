import constants
from train_gans import *

def train1():
  # size of the latent space
  latent_dim = 100
  # create the discriminator
  d_model = define_discriminator()
  # create the generator
  g_model = define_generator(latent_dim)
  # create the gan
  gan_model = define_gan(g_model, d_model)
  # load image data
  dataset = load_real_samples()
  # train model
  train(g_model, d_model, gan_model, dataset, latent_dim)

  # save models
  mn = 'g_model'
  dump(g_model, constants.ROOT + '/results/models/' + mn + '.joblib')

# generate imgs
def predict1():
  # load model
  model = load(constants.ROOT + '/results/models/' + mn + '.joblib')
  # generate images
  latent_points = generate_latent_points(100, 25)
  # generate images
  X = model.predict(latent_points)
  # plot the result
  save_plot(X, 5)

train1()
predict1()