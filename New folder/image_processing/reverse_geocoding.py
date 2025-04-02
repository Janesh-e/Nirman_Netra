from geopy.geocoders import Nominatim
import rasterio
import numpy as np
from rasterio.warp import transform
from rasterio.warp import reproject, Resampling
from rasterio.transform import Affine



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
    cx_cy_pairs = []
    for box in bounding_boxes:
        print(type(box))
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        # Calculate the central pixel coordinates of the bounding box
        cx = x + w // 2
        cy = y + h // 2
        cx_cy_pairs.append((cx,cy))
        with rasterio.open(image_path) as dataset:
            # Use the dataset's transform to convert pixel coordinates to geographical coordinates
            lon,lat = dataset.xy(cy, cx)
            lat_long_pairs.append((lat,lon))
    return lat_long_pairs, cx_cy_pairs

def coordinates_to_address(lat_long_pairs):
    addresses = []
    geolocator = Nominatim(user_agent="ekambaramjanesh@gmail.com")
    i = 0
    for (lat,long) in lat_long_pairs:
        location = geolocator.reverse((lat, long), exactly_one=True)
        if location:
            addresses.append(location.address)
        else:
            addresses.append("Address not found")
        i = i+1
        if(i==15):break
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


def calculateHeight(imagepath,latitude,longitude):
    with rasterio.open(imagepath) as src:
        elevation_data = src.read(1)
    flat_data = elevation_data.flatten()
    filtered_data = flat_data[flat_data != 255]
    unique_values, counts = np.unique(filtered_data, return_counts=True)
    most_frequent_index = np.argmax(counts)
    most_frequent_value = unique_values[most_frequent_index]
    frequency = counts[most_frequent_index]
    print("the base elevation is ", most_frequent_value)
    
    ds = rasterio.open(imagepath)
    dem_data = ds.read(1)
    dem_crs = ds.crs
    #print(dem_crs)
    if dem_crs.to_string() != 'EPSG:4326':
        x, y = transform('EPSG:4326', dem_crs, [longitude], [latitude])
        x, y = x[0], y[0]
    else:
        x, y = longitude, latitude
    row, col = ds.index(x, y)
    if 0 <= row < ds.height and 0 <= col < ds.width:
        elevation = ds.read(1)[row, col]
        if(elevation<most_frequent_value):
            arr=[0,0]
            return arr
        height= elevation-most_frequent_value
        if(height<=15):
            arr=[0,0]
            return arr
        arr=[height,height//10]
        return arr
        #print("the building elevation is ",elevation)
        #print("the height is ",height,"foot")
        #print("the no of floors is ",height//10)
    else:
        return [-1,-1]
    

def resize_geotiff(input_path, output_path, desired_width, desired_height):
    """
    Resize a GeoTIFF file to the desired width and height while preserving its metadata.
    Args:
        input_path (str): Path to the input GeoTIFF file.
        output_path (str): Path to save the resized GeoTIFF file.
        desired_width (int): Desired width of the output GeoTIFF.
        desired_height (int): Desired height of the output GeoTIFF.
    """
    with rasterio.open(input_path) as src:
        # Get the original metadata
        meta = src.meta.copy()

        # Calculate the scaling factors
        width_scale = desired_width / src.width
        height_scale = desired_height / src.height

        # Update the metadata with the new dimensions and transform
        meta.update({
            'width': desired_width,
            'height': desired_height,
            'transform': Affine.scale(width_scale, height_scale) * src.transform
        })

        # Create the output file
        with rasterio.open(output_path, 'w', **meta) as dst:
            # Reproject the data to the new dimensions
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=meta['transform'],
                    dst_crs=src.crs,
                    resampling=Resampling.bilinear
                )