import cv2
import os
import datetime

class ImageSaver:

    # -- Settings -- #
    DEFAULT_FOLDER_NAME = "PhotoSaves"

    def __init__(self, default_folder=None):
        """
        Initializes the ImageSaver with a default folder.

        Args:
            default_folder (str): The default folder to save images. If None, uses the Photos folder in Documents.
        """
        self.default_folder = default_folder or self.get_default_folder()

    def get_default_folder(self):
        """
        Returns the path to the \"Photos\" folder in the user's Documents folder in a cross-platform way.
        """
        home_directory = os.path.expanduser("~")
        documents_directory = os.path.join(home_directory, "Documents")
        default_directory = os.path.join(documents_directory, self.DEFAULT_FOLDER_NAME)
        print(f"Default image save directory: {default_directory}")
        # Return the default directory.
        return default_directory
         
    def save_image(self, image, filename, folder=None):
        """
        Saves an image to the specified folder or the default folder.

        Args:
            image (numpy.ndarray): The image to save.
            filename (str): The name of the file (e.g., 'photo.jpg').
            folder (str): The folder to save the image. If None, uses the default folder.

        Returns:
            str: The full path to the saved image.
        """
        save_folder = folder or self.default_folder

        # Ensure the folder exists
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        
        # Get the current system time.
        now = datetime.datetime.now()
        formatted_time = now.strftime("%B %d, %Y %H_%M_%S")

        # Construct the file path, adding the current time to the file name to ensure a unique name.
        file_path = os.path.join(save_folder, filename + formatted_time + ".png")

        # Save the image
        cv2.imwrite(file_path, image)
        print(f"Image saved to {file_path}")

        return file_path