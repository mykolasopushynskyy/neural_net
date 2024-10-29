from PIL import Image
import os


def resize_image(input_image_path, output_image_path, size=(28, 28), resample=Image.Resampling.LANCZOS):
    # Open an image file
    with Image.open(input_image_path) as img:
        # Resize the image with the specified resampling filter
        resized_img = img.resize(size, reducing_gap=3.0, resample=resample)
        # Save the resized image as PNG
        resized_img.save(output_image_path, format='PNG')
        print(f"Image resized and saved to {output_image_path}")


def resize_images_in_folder(in_folder, out_folder, size=(28, 28), resample=Image.Resampling.LANCZOS):
    # Ensure the output folder exists
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # Process each image in the input folder
    for filename in os.listdir(in_folder):
        # Construct the full file path
        input_image_path = os.path.join(in_folder, filename)

        # Check if it's an image file (basic check based on file extension)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # Construct output path with PNG extension
            output_image_path = os.path.join(out_folder, f"{os.path.splitext(filename)[0]}.png")

            # Resize and save the image
            resize_image(input_image_path, output_image_path, size, resample)

if __name__ == "__main__":

    input_folder = '.'  # Folder containing original images
    output_folder = './png'  # Folder to save resized images

    resize_images_in_folder(input_folder, output_folder, resample=Image.Resampling.LANCZOS)
