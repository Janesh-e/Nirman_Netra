from geopy.geocoders import Nominatim
import rasterio

# function to display the coordinates of
# of the points clicked on the image
'''def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + ',' +
                    str(y), (x, y), font,
                    1, (255, 0, 0), 2)
        cv2.imshow('image', img)

'''
def pixel_to_coordinates(bounding_boxes, image_path):
    lat_long_pairs = []
    print(bounding_boxes)
    for box in bounding_boxes:
        print(type(box))
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        # Calculate the central pixel coordinates of the bounding box
        cx = x + w // 2
        cy = y + h // 2
        with rasterio.open(image_path) as dataset:
            # Use the dataset's transform to convert pixel coordinates to geographical coordinates
            lon,lat = dataset.xy(cy, cx)
            lat_long_pairs.append((lat,lon))
    return lat_long_pairs

def coordinates_to_address(lat_long_pairs):
    addresses = []
    geolocator = Nominatim(user_agent="rrithika201@gmail.com")
    for (lat,long) in lat_long_pairs:
        location = geolocator.reverse((lat, long), exactly_one=True)
        if location:
            addresses.append(location.address)
        else:
            addresses.append("Address not found")
    return addresses


# driver function
'''if __name__ == "__main__":
    # reading the image
    img = cv2.imread("C:/Rithika_Folder/SIH_2024/sairam_orthophoto.png", 1)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)

    # displaying the image
    cv2.imshow('image', img)

    # setting mouse handler for the image
    # and calling the click_event() function
    cv2.setMouseCallback('image', click_event)

    # wait for a key to be pressed to exit
    cv2.waitKey(0)

    # close the window
    cv2.destroyAllWindows()'''



def save_processed_image_with_geotags(original_image_path, predicted_mask_image, output_image_path):
    # Open the original image to read metadata
    with rasterio.open(original_image_path) as src:
        # Get the metadata of the original image
        meta = src.meta.copy()

        # Set the dtype of the output mask (this should match your processed mask)
        meta.update(dtype='uint8')  # Update metadata (change as needed for your mask)

        # Save the processed image with the same metadata as the original image
        with rasterio.open(output_image_path, 'w', **meta) as dst:
            dst.write(predicted_mask_image[0], 1)  # Red channel
            dst.write(predicted_mask_image[1], 2)  # Green channel
            dst.write(predicted_mask_image[2], 3)  # Blue channel