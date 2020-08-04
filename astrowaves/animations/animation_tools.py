import cv2
import numpy as np
import glob
import tqdm
from cv2 import VideoWriter, VideoWriter_fourcc
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os


def visualize_waves(
        waves, output_path="C:\\Users\\Wojtek\\Documents\\Doktorat\\AstrocyteCalciumWaveDetector\\debug",
        filename='waves.mp4'):

    ims = []

    fig = plt.figure()

    for i in tqdm.trange(waves.shape[2]):
        im = plt.imshow(waves[:, :, i], animated=True, cmap='gray', vmin=0, vmax=255)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                    repeat_delay=1000)

    ani.save(os.path.join(output_path, filename))


def create_timespace_wave_video(waves, timespace, output_video_path, filename):

    if isinstance(waves, str):
        waves = np.load(waves)
        waves = waves.astype('uint8')
        waves[waves > 0] = 255
    elif waves is not np.ndarray:
        print("Error with argument waves")
    if isinstance(timespace, str):
        timespace = np.load(timespace)
    elif timespace is not np.ndarray:
        print("Error with argument timespace")

    output_sequence = np.concatenate((timespace, waves), axis=0)

    ims = []

    fig = plt.figure()

    for i in tqdm.trange(output_sequence.shape[2]):
        im = plt.imshow(output_sequence[:, :, i], animated=True, cmap='gray', vmin=0, vmax=255)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                    repeat_delay=1000)

    ani.save(os.path.join(output_video_path, filename))

    # output_sequence = waves  # np.concatenate((timespace, waves), axis=0)
    # video_size = output_sequence.shape[:-1]

    # FPS=24
    # seconds=50
    # fourcc = VideoWriter_fourcc(*'MP42')
    # video = VideoWriter(output_video_path, fourcc, float(FPS), (output_sequence.shape[0], output_sequence.shape[1]))

    # for i in tqdm.trange(output_sequence.shape[-1]):
    #     slic = output_sequence[:,:,i].astype('uint8')
    #     slic_3d = np.zeros(shape=(slic.shape[0], slic.shape[1], 3))
    #     slic_3d[:,:, 0] = slic
    #     slic_3d[:,:, 1] = slic
    #     slic_3d[:,:, 2] = slic
    #     print(slic_3d.shape)
    #     video.write(slic)
    # video.release()


if __name__ == '__main__':

    waves = r'C:\Users\Wojtek\Documents\Doktorat\Astral\data\Cont_AA_2_4\\waves_morph.npy'
    timespace = r"C:\Users\Wojtek\Documents\Doktorat\Astral\data\Cont_AA_2_4\\timespace.npy"
    output_path = r"C:\Users\Wojtek\Documents\Doktorat\Astral\data\Cont_AA_2_4\\video.mp4"

    create_timespace_wave_video(
        waves, timespace, r'C:\Users\Wojtek\Documents\Doktorat\Astral\data\Cont_AA_2_4', 'video.mp4')
