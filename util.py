import os
import re
import numpy
from PIL import Image
import matplotlib.pyplot as plt

def read_images(input_folder, grayscale=True):
    image_arrays = []

    # Process each image in the input folder
    for filename in os.listdir(input_folder):
        # Construct the full file path
        input_image_path = os.path.join(input_folder, filename)

        # Check if it's an image file (basic check based on file extension)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # Open the image
            with Image.open(input_image_path) as img:
                # Convert to grayscale if needed
                if grayscale:
                    img = img.convert('L')  # 'L' mode is for grayscale

                # Convert the image to a NumPy array
                img_array = numpy.array(img)

                # Flatten the image into a linear array
                img_flattened = img_array.flatten()

                # Append the flattened array to the list
                label = re.findall(r'\d+', filename)[1]
                image_arrays.append(dict(label=int(label), data=img_flattened))

    # Convert list of arrays into a single 2D NumPy array
    return numpy.array(image_arrays)

def save_grayscale_image(grayscale_data, output_image_path):
    # Create the directory if it doesn't exist
    directory = os.path.dirname(output_image_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    # Check if the input grayscale data matches the required size (28x28)
    if grayscale_data.shape != (28, 28):
        print("The grayscale data must have a shape of 28x28.")
        raise ValueError("The grayscale data must have a shape of 28x28.")

    # Create a grayscale image using PIL
    img = Image.fromarray(grayscale_data, 'L')  # 'L' mode is for grayscale

    # Save the image
    img.save(output_image_path, format='PNG')
    print(f"Grayscale image saved to {output_image_path}")


def save_plot(x, y, plot_path='plot.png', xlabel='x-axis', ylabel='y-axis', title='Plot', grid=True):
    """
    Generates a plot for given x and y arrays and saves it as an image file.

    Parameters:
    - x (array-like): Array of x values.
    - y (array-like): Array of y values.
    - plot_path (str): The file path (including filename) to save the plot as (default is 'plot.png').
    - xlabel (str): Label for the x-axis (default is 'x-axis').
    - ylabel (str): Label for the y-axis (default is 'y-axis').
    - title (str): Title of the plot (default is 'Plot').
    - grid (bool): Whether to display a grid on the plot (default is True).
    """
    # Create the directory if it doesn't exist
    directory = os.path.dirname(plot_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, color='blue', linewidth=2)

    # Add labels, title, and grid
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if grid:
        plt.grid(visible=True, linestyle='--', color='gray', alpha=0.7)

    # Save the plot as a PNG file
    plt.savefig(plot_path, format='png', dpi=300)
    plt.close()  # Close the plot to free memory