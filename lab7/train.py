import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from IPython import display
import matplotlib.pyplot as plt

def real_data_target(size):
    '''
    Tensor containing ones, with shape = size
    '''
    data = torch.ones(size, 1)
    data = data.to(device)
    return data

def fake_data_target(size):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = torch.zeros(size, 1)
    data = data.to(device)
    return data

def train_discriminator(discriminator,optimizer, real_data, fake_data,loss):
    # Reset gradients
    optimizer.zero_grad()
    
    # Propagate real data
    prediction_real = discriminator(real_data)
    error_real = loss(prediction_real, real_data_target(real_data.size(0)))
    error_real.backward()

    # Propagate fake data
    prediction_fake = discriminator(fake_data)
    error_fake = loss(prediction_fake, fake_data_target(real_data.size(0)))
    error_fake.backward()
    
    # Take a step
    optimizer.step()
    
    # Return error
    return error_real + error_fake, prediction_real, prediction_fake

def plt_output(fake_data):
    plt.figure(figsize=(8,8))
    plt.xlim(-20,20)
    plt.ylim(-20,20)
    plt.scatter(fake_data[:,0],fake_data[:,1])
    plt.show()

def train_generator(discriminator,optimizer, fake_data,loss):
    # Reset gradients
    optimizer.zero_grad()

    # Propagate the fake data through the discriminator and backpropagate.
    # Note that since we want the generator to output something that gets
    # the discriminator to output a 1, we use the real data target here.
    prediction = discriminator(fake_data)
    error = loss(prediction, real_data_target(prediction.size(0)))
    error.backward()
    
    # Update weights with gradients
    optimizer.step()
    
    # Return error
    return error

def trainSnake(data_loader,num_epochs,generator,discriminator,d_optimizer,g_optimizer,loss,num_test_samples,Logger):
    from models import snakeGan as snk
    logger = Logger(model_name='VGAN', data_name='Snake')
    num_batches = len(data_loader)
    test_noise = snk.noise(num_test_samples)
    for epoch in range(num_epochs):
        for n_batch, (real_batch) in enumerate(data_loader):
            real_data = real_batch
            # Train discriminator on a real batch and a fake batch
            #real_data = vanil.images_to_vectors(real_batch)
            real_data = real_data.to(device)
            fake_data = generator(snk.noise(real_data.size(0))).detach()
            d_error, d_pred_real, d_pred_fake = train_discriminator(discriminator,d_optimizer,
                                                                    real_data, fake_data,loss)
            
            # Train generator

            fake_data = generator(snk.noise(real_batch.size(0)))
            g_error = train_generator(discriminator,g_optimizer, fake_data,loss)
            
            # Log errors and display progress

            logger.log(d_error, g_error, epoch, n_batch, num_batches)
            if (n_batch) % 100 == 0:
                display.clear_output(True)
                # Display Images
                test_plot = plt_output(generator(test_noise).cpu().detach().numpy())
                #test_images = vanil.vectors_to_images(generator(test_noise)).data.cpu()
                #logger.log_images(test_images, num_test_samples, epoch, n_batch, num_batches);
                # Display status Logs
                logger.display_status(
                    epoch, num_epochs, n_batch, num_batches,
                    d_error, g_error, d_pred_real, d_pred_fake
                )
                
            # Save model checkpoints
            logger.save_models(generator, discriminator, epoch)

def trainMnist(data_loader,num_epochs,generator,discriminator,d_optimizer,g_optimizer,loss,num_test_samples,Logger):
    from models import mnistGan as mgan
    logger = Logger(model_name='VGAN', data_name='Mnist')
    num_batches = len(data_loader)
    test_noise = mgan.noise(num_test_samples)
    for epoch in range(num_epochs):
        for n_batch, (real_batch,_) in enumerate(data_loader):
            real_data = real_batch
            # Train discriminator on a real batch and a fake batch
            real_data = mgan.images_to_vectors(real_batch)
            real_data = real_data.to(device)
            fake_data = generator(mgan.noise(real_data.size(0))).detach()
            d_error, d_pred_real, d_pred_fake = train_discriminator(discriminator,d_optimizer,
                                                                    real_data, fake_data,loss)
            
            # Train generator

            fake_data = generator(mgan.noise(real_batch.size(0)))
            g_error = train_generator(discriminator,g_optimizer, fake_data,loss)
            
            # Log errors and display progress

            logger.log(d_error, g_error, epoch, n_batch, num_batches)
            if (n_batch) % 100 == 0:
                display.clear_output(True)
                # Display Images
                test_plot = plt_output(generator(test_noise).cpu().detach().numpy())
                test_images = mgan.vectors_to_images(generator(test_noise)).data.cpu()
                logger.log_images(test_images, num_test_samples, epoch, n_batch, num_batches);
                # Display status Logs
                logger.display_status(
                    epoch, num_epochs, n_batch, num_batches,
                    d_error, g_error, d_pred_real, d_pred_fake
                )
                
            # Save model checkpoints
            logger.save_models(generator, discriminator, epoch)

def trainDC(data_loader,num_epochs,generator,discriminator,d_optimizer,g_optimizer,loss,num_test_samples,Logger):
    from models import dcGan as dc
    num_test_samples = num_test_samples
    test_noise = dc.noise(num_test_samples)
    num_batches = len(data_loader)
    logger = Logger(model_name='DCGAN', data_name='CelebFace')

    for epoch in range(num_epochs):
        for n_batch, (real_data,_) in enumerate(data_loader):

            # Train Discriminator
            
            real_data = real_data.to(device)
            fake_data = generator(dc.noise(real_data.size(0))).detach()
            d_error, d_pred_real, d_pred_fake = train_discriminator(discriminator,d_optimizer,
                                                                    real_data, fake_data,loss)

            # Train Generator
            
            fake_data = generator(dc.noise(real_data.size(0)))
            g_error = train_generator(discriminator,g_optimizer, fake_data,loss)

            # Log error and display progress
            logger.log(d_error, g_error, epoch, n_batch, num_batches)
            if (n_batch) % 100 == 0:
                display.clear_output(True)
                # Display Images
                test_images = generator(test_noise).data.cpu()
                logger.log_images(test_images, num_test_samples, epoch, n_batch, num_batches);
                # Display status Logs
                logger.display_status(
                    epoch, num_epochs, n_batch, num_batches,
                    d_error, g_error, d_pred_real, d_pred_fake
                )

            # Save model checkpoints
            logger.save_models(generator, discriminator, epoch)

def trainCifa(data_loader,num_epochs,generator,discriminator,d_optimizer,g_optimizer,loss,num_test_samples,Logger):
    from models import dcGan as dc
    num_test_samples = num_test_samples
    test_noise = dc.noise(num_test_samples)
    num_batches = len(data_loader)
    logger = Logger(model_name='DCGAN', data_name='Cifar10')

    for epoch in range(num_epochs):
        for n_batch, (real_data,_) in enumerate(data_loader):

            # Train Discriminator
            
            real_data = real_data.to(device)
            fake_data = generator(dc.noise(real_data.size(0))).detach()
            d_error, d_pred_real, d_pred_fake = train_discriminator(discriminator,d_optimizer,
                                                                    real_data, fake_data,loss)

            # Train Generator
            
            fake_data = generator(dc.noise(real_data.size(0)))
            g_error = train_generator(discriminator,g_optimizer, fake_data,loss)

            # Log error and display progress
            logger.log(d_error, g_error, epoch, n_batch, num_batches)
            if (n_batch) % 100 == 0:
                display.clear_output(True)
                # Display Images
                test_images = generator(test_noise).data.cpu()
                logger.log_images(test_images, num_test_samples, epoch, n_batch, num_batches);
                # Display status Logs
                logger.display_status(
                    epoch, num_epochs, n_batch, num_batches,
                    d_error, g_error, d_pred_real, d_pred_fake
                )

            # Save model checkpoints
            logger.save_models(generator, discriminator, epoch)