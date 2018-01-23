
import os
import cv2
import data_manager

_LIST_BBOX_NAME = "list_bbox.txt"
_LIST_LABEL_NAME = "list_category_img.txt"
_LABEL_COUNT_NAME = 'label_count'
_CROP_IMAGE_PATH = '../cropImage/'


def crop_image():
    bbox_file = open(_LIST_BBOX_NAME)
    label_file = open(_LIST_LABEL_NAME)

    bbox_list = bbox_file.readlines()
    label_list = label_file.readlines()

    label_dic_count = {}

    for line_index, bbox_file_line in enumerate(bbox_list[2:]):
        bbox_line_content = bbox_file_line.strip().split()
        label_line_content = label_list[line_index + 2].strip().split()

        image_name = bbox_line_content[0]
        x_min = int(bbox_line_content[1])
        y_min = int(bbox_line_content[2])
        x_max = int(bbox_line_content[3])
        y_max = int(bbox_line_content[4])
        label = str(label_line_content[1])

        if not os.path.exists(label):
            os.mkdir(label)

        path = os.path.join('../', image_name)
        image = cv2.imread(path)

        if image is not None:
            crop_image = image[y_min:y_max, x_min:x_max]

            new_image_name = os.path.join(label, "%d.jpg" % (line_index + 1))

            cv2.imwrite(new_image_name, crop_image)
        else:
            print("%d img is not exits" % (line_index + 1))

        count = label_dic_count.get(label)

        if not count:
            label_dic_count[label] = 1
        else:
            count += 1
            label_dic_count[label] = count

    bbox_file.close()
    label_file.close()
    data_manager.write_data_to_json(_LABEL_COUNT_NAME, label_dic_count)

def enhance_image():
    count_data = data_manager.read_data_from_json(_LABEL_COUNT_NAME)
    for k, v in count_data.items():
        if v < 1000:
            blur_rotate_iamge(k)

def blur_rotate_iamge(label):
    path = os.path.join(_CROP_IMAGE_PATH, label + '/')

    for file_names in os.listdir(path):
        file_path = path + file_names
        img = cv2.imread(file_path)

        if not img is None and 'blur' not in file_names:
            path_params = file_names.split('.')

            blur = cv2.blur(img, (3, 3))
            new_path = path + path_params[0] + '_blur.jpg'
            cv2.imwrite(new_path, blur)

            rotate = cv2.flip(img, 90)
            new_path = path + path_params[0] + '_rotate.jpg'
            cv2.imwrite(new_path, rotate)

if __name__ == '__main__':
    # crop_image()
    enhance_image()