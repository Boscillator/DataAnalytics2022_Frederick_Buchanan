from FirePrediction.data_loader import FireSequence
import numpy as np

from FirePrediction.util import ensure_exists

BATCH_SIZE = 100
def main():
    data = FireSequence('data/processed/enriched', BATCH_SIZE)

    total_pixels_on_fire = 0
    total_pixels_spread = 0
    total_pixels = 0
    avg_temps = []
    max_temps = []
    min_temps = []

    avg_spread = []

    for i in range(len(data)):
        Xs, ys = data[i]

        for j in range(Xs.shape[0]):
            today = Xs[j, :, :, 0]
            tomorrow = ys[j, :, :]
            spread = tomorrow - today
            temperature = Xs[j,4,:,:]

            total_pixels_on_fire += np.sum(today)
            total_pixels_spread += np.sum(spread)
            total_pixels += np.prod(today.shape)

            avg_spread.append(np.sum(spread))
            avg_temps.append(np.mean(temperature))
            max_temps.append(np.max(temperature))
            min_temps.append(np.min(temperature))

    total_pixels_not_on_fire = total_pixels - total_pixels_on_fire
    p_pixel_changes_from_not_on_fire_to_on_fire = total_pixels_spread / total_pixels_not_on_fire
    avg_temps = np.mean(avg_temps)
    max_temps = np.max(max_temps)
    min_temps = np.min(min_temps)

    avg_spread = np.mean(avg_spread)

    ensure_exists('plots/eda')
    with open('plots/eda/eda.txt', 'w') as f:
        f.write(f"Total pixels on fire: {total_pixels_on_fire:,}\n")
        f.write(f"Total pixels spread: {total_pixels_spread:,}\n")
        f.write(f"Total pixels: {total_pixels:,}\n")
        f.write(f"Total pixels not on fire: {total_pixels_not_on_fire:,}\n")
        f.write(f"Average temperature: {avg_temps}\n")
        f.write(f"Minimum temperature: {min_temps}\n")
        f.write(f"Maximum temperature: {max_temps}\n")
        f.write(f"Probability a pixel change from not on fire to on fire: {p_pixel_changes_from_not_on_fire_to_on_fire * 100:.2f}%\n")
        f.write(f"Average Spread: {avg_spread}\n")

if __name__ == '__main__':
    main()
