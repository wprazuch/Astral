import numpy as np
import os
import astrowaves.animations.animation_tools as anim_tools

if __name__ == "__main__":

    debug_path = 'C:\\Users\\Wojtek\\Documents\\Doktorat\\AstrocyteCalciumWaveDetector\\debug'

    waves = np.load(os.path.join(debug_path, 'waves_morph.npy'))
    timespace = np.load(os.path.join(debug_path, 'timespace.npy'))

    anim_tools.create_timespace_wave_video(waves, timespace, output_video_path=debug_path, filename="result_std1.mp4")