import numpy as np
import tensorflow as tf
from tensorflow import keras

'''
(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()
print(train_images[0].tolist())
print('\n\n\n\n\n')
print(train_labels[0])
print(train_images[0].shape)
print(train_labels[0].shape)
print(train_images.shape)
print(train_labels.shape)
'''

def load_data():
    image = [[[59, 62, 63], [43, 46, 45], [50, 48, 43], [68, 54, 42], [98, 73, 52], [119, 91, 63], [139, 107, 75], [145, 110, 80], [149, 117, 89], [149, 120, 93], [131, 103, 77], [125, 99, 76], [142, 115, 91], [144, 112, 86], [137, 105, 79], [129, 97, 71], [137, 106, 79], [134, 106, 76], [124, 97, 64], [139, 113, 78], [139, 112, 75], [133, 105, 69], [136, 105, 74], [139, 108, 77], [152, 120, 89], [163, 131, 100], [168, 136, 108], [159, 129, 102], [158, 130, 104], [158, 132, 108], [152, 125, 102], [148, 124, 103]], [[16, 20, 20], [0, 0, 0], [18, 8, 0], [51, 27, 8], [88, 51, 21], [120, 82, 43], [128, 89, 45], [127, 86, 44], [126, 87, 50], [116, 79, 44], [106, 70, 37], [101, 67, 35], [105, 70, 36], [113, 74, 35], [109, 70, 33], [112, 72, 37], [119, 79, 44], [109, 71, 33], [105, 69, 27], [125, 89, 46], [127, 92, 46], [122, 85, 39], [131, 89, 47], [124, 82, 41], [121, 79, 37], [131, 89, 48], [132, 91, 53], [133, 94, 58], [133, 96, 60], [123, 88, 55], [119, 83, 50], [122, 87, 57]], [[25, 24, 21], [16, 7, 0], [49, 27, 8], [83, 50, 23], [110, 72, 41], [129, 92, 54], [130, 93, 55], [121, 82, 47], [113, 77, 43], [112, 78, 44], [112, 79, 46], [106, 75, 45], [105, 73, 38], [128, 92, 48], [124, 87, 47], [130, 92, 56], [127, 89, 56], [122, 85, 51], [115, 79, 43], [120, 85, 47], [130, 95, 54], [131, 96, 55], [139, 102, 62], [127, 90, 51], [126, 89, 49], [127, 89, 50], [130, 92, 53], [142, 105, 68], [130, 94, 58], [118, 84, 50], [120, 84, 50], [109, 73, 42]], [[33, 25, 17], [38, 20, 4], [87, 54, 25], [106, 63, 28], [115, 70, 33], [117, 74, 35], [114, 72, 37], [105, 62, 33], [107, 68, 33], [121, 84, 45], [125, 90, 53], [109, 75, 40], [113, 77, 38], [146, 105, 58], [133, 91, 47], [127, 84, 45], [118, 76, 40], [117, 76, 41], [127, 87, 52], [122, 81, 43], [132, 92, 51], [137, 99, 58], [136, 99, 57], [131, 93, 52], [124, 86, 44], [130, 91, 50], [132, 90, 49], [135, 93, 51], [130, 90, 50], [125, 87, 50], [121, 85, 48], [94, 62, 35]], [[50, 32, 21], [59, 32, 11], [102, 65, 34], [127, 79, 39], [124, 77, 36], [121, 77, 36], [120, 78, 40], [114, 74, 39], [107, 72, 34], [125, 88, 49], [129, 89, 51], [106, 68, 31], [108, 71, 33], [124, 83, 42], [121, 78, 39], [108, 68, 29], [98, 65, 23], [110, 74, 37], [117, 80, 49], [120, 80, 41], [134, 93, 50], [140, 106, 66], [131, 95, 58], [141, 98, 66], [135, 92, 51], [127, 84, 45], [121, 79, 41], [119, 79, 40], [103, 67, 32], [87, 57, 27], [75, 47, 23], [67, 42, 25]], [[71, 48, 29], [84, 53, 24], [110, 73, 37], [129, 82, 38], [136, 88, 45], [131, 84, 42], [129, 84, 43], [119, 77, 37], [108, 70, 33], [122, 82, 44], [123, 81, 39], [105, 65, 25], [107, 72, 31], [111, 77, 31], [108, 74, 34], [98, 65, 27], [94, 62, 21], [97, 63, 32], [83, 56, 38], [88, 58, 36], [102, 68, 42], [97, 69, 46], [88, 54, 36], [118, 74, 72], [140, 96, 79], [136, 97, 64], [120, 80, 45], [107, 68, 34], [88, 54, 24], [67, 39, 15], [35, 10, 0], [32, 13, 4]], [[97, 69, 40], [111, 75, 36], [123, 85, 43], [130, 84, 38], [136, 88, 44], [132, 83, 40], [122, 74, 30], [121, 74, 31], [127, 83, 46], [138, 94, 54], [124, 79, 34], [120, 79, 39], [107, 71, 34], [80, 50, 14], [68, 43, 17], [74, 41, 17], [101, 51, 21], [105, 56, 23], [65, 37, 16], [58, 36, 19], [63, 37, 18], [78, 51, 31], [136, 93, 83], [122, 68, 80], [139, 86, 79], [151, 106, 69], [129, 87, 49], [108, 68, 36], [95, 59, 29], [96, 63, 37], [89, 61, 38], [66, 47, 30]], [[115, 82, 49], [119, 76, 33], [130, 90, 47], [140, 97, 53], [133, 88, 48], [127, 81, 40], [138, 90, 47], [137, 89, 46], [131, 86, 48], [133, 89, 46], [134, 91, 46], [108, 70, 39], [72, 39, 19], [51, 26, 10], [41, 22, 14], [72, 31, 17], [181, 102, 69], [209, 127, 81], [125, 76, 47], [68, 40, 23], [64, 38, 17], [82, 53, 30], [123, 77, 62], [112, 56, 55], [135, 81, 60], [151, 103, 61], [137, 95, 54], [114, 76, 39], [105, 69, 34], [101, 66, 33], [126, 92, 59], [102, 74, 46]], [[137, 100, 68], [128, 82, 41], [132, 91, 51], [128, 87, 48], [119, 81, 44], [123, 82, 43], [128, 85, 44], [130, 85, 44], [121, 80, 40], [137, 97, 54], [131, 94, 53], [74, 42, 20], [54, 25, 16], [50, 29, 16], [44, 29, 18], [86, 39, 15], [203, 106, 56], [217, 109, 62], [162, 90, 71], [100, 58, 49], [77, 42, 27], [75, 43, 24], [74, 39, 24], [76, 35, 22], [107, 67, 36], [135, 96, 59], [135, 97, 58], [129, 91, 49], [127, 89, 48], [119, 83, 43], [125, 86, 45], [134, 95, 56]], [[154, 120, 89], [154, 112, 77], [156, 114, 82], [140, 100, 65], [123, 89, 53], [125, 86, 50], [126, 86, 48], [127, 91, 52], [133, 97, 60], [132, 97, 68], [90, 60, 30], [63, 35, 9], [62, 33, 16], [70, 39, 20], [79, 50, 30], [103, 53, 26], [152, 70, 33], [148, 64, 37], [141, 79, 61], [121, 75, 57], [101, 58, 41], [96, 54, 33], [86, 48, 24], [75, 38, 21], [101, 63, 32], [136, 91, 53], [136, 92, 53], [134, 93, 50], [133, 93, 52], [132, 93, 52], [128, 86, 45], [133, 92, 55]], [[154, 122, 94], [155, 117, 82], [156, 117, 82], [147, 108, 70], [133, 100, 64], [137, 100, 66], [139, 102, 68], [134, 102, 66], [141, 111, 81], [121, 87, 68], [80, 40, 13], [97, 53, 17], [90, 45, 17], [98, 56, 30], [137, 91, 57], [139, 84, 49], [148, 87, 54], [134, 73, 37], [138, 82, 46], [134, 85, 57], [140, 92, 76], [175, 129, 106], [142, 99, 53], [102, 61, 26], [108, 67, 25], [135, 90, 41], [131, 87, 45], [133, 91, 51], [138, 97, 57], [136, 95, 55], [130, 86, 46], [134, 93, 57]], [[145, 114, 89], [146, 109, 73], [146, 109, 69], [135, 97, 55], [127, 92, 57], [129, 94, 65], [117, 84, 55], [103, 74, 42], [130, 103, 70], [120, 83, 55], [111, 60, 14], [146, 86, 22], [136, 78, 23], [163, 116, 77], [169, 115, 69], [152, 100, 52], [161, 116, 73], [148, 97, 57], [177, 121, 82], [161, 110, 71], [195, 150, 113], [209, 167, 123], [189, 146, 94], [125, 78, 40], [108, 63, 25], [140, 96, 52], [137, 95, 59], [132, 93, 56], [136, 95, 57], [133, 90, 51], [132, 87, 46], [133, 92, 56]], [[142, 115, 86], [141, 106, 69], [140, 105, 68], [144, 105, 64], [147, 110, 74], [121, 89, 65], [84, 56, 34], [88, 61, 33], [109, 80, 44], [101, 57, 23], [138, 79, 19], [213, 150, 59], [178, 123, 41], [191, 150, 98], [211, 169, 122], [189, 148, 99], [205, 164, 110], [207, 162, 115], [213, 164, 118], [191, 143, 91], [199, 158, 97], [188, 151, 88], [161, 121, 76], [130, 83, 50], [124, 77, 38], [131, 87, 51], [130, 91, 61], [131, 93, 60], [134, 93, 57], [135, 91, 52], [136, 89, 48], [133, 91, 56]], [[158, 131, 98], [154, 119, 82], [142, 107, 74], [143, 102, 65], [132, 92, 59], [90, 59, 36], [72, 44, 22], [81, 52, 24], [84, 47, 19], [107, 55, 25], [165, 106, 50], [229, 176, 92], [183, 137, 57], [191, 158, 103], [239, 216, 176], [219, 192, 149], [228, 188, 128], [225, 188, 120], [214, 177, 112], [216, 174, 112], [210, 171, 110], [200, 169, 109], [189, 162, 114], [174, 137, 100], [161, 118, 76], [139, 95, 57], [134, 96, 66], [126, 90, 59], [131, 92, 56], [142, 98, 60], [136, 89, 48], [138, 97, 61]], [[145, 115, 79], [149, 109, 66], [147, 108, 68], [147, 105, 65], [136, 95, 62], [80, 47, 21], [89, 57, 32], [105, 68, 40], [96, 51, 26], [129, 81, 45], [192, 152, 113], [185, 148, 107], [145, 101, 51], [203, 162, 121], [223, 200, 170], [242, 227, 196], [244, 227, 186], [238, 220, 165], [241, 219, 163], [227, 197, 144], [225, 191, 139], [235, 209, 157], [219, 206, 164], [224, 208, 181], [215, 192, 156], [156, 118, 78], [128, 89, 57], [129, 95, 62], [131, 95, 60], [133, 97, 60], [128, 89, 50], [130, 92, 56]], [[148, 116, 79], [146, 100, 54], [145, 100, 55], [147, 100, 51], [133, 96, 54], [63, 42, 21], [66, 43, 31], [88, 50, 34], [113, 65, 37], [182, 146, 110], [220, 191, 169], [138, 94, 71], [162, 105, 63], [206, 156, 112], [196, 166, 135], [247, 234, 212], [255, 253, 232], [255, 252, 219], [245, 234, 197], [236, 217, 180], [230, 208, 170], [215, 196, 160], [231, 217, 197], [250, 241, 229], [241, 229, 195], [158, 132, 78], [125, 95, 49], [126, 97, 58], [124, 92, 52], [125, 91, 49], [126, 88, 46], [124, 88, 54]], [[149, 115, 79], [143, 95, 49], [144, 97, 51], [151, 99, 51], [132, 87, 49], [64, 40, 21], [84, 59, 41], [112, 69, 37], [163, 121, 75], [223, 204, 166], [206, 182, 157], [145, 90, 56], [196, 133, 84], [204, 157, 110], [220, 188, 156], [243, 226, 208], [245, 237, 226], [239, 233, 215], [234, 224, 201], [231, 217, 192], [195, 181, 152], [150, 137, 100], [208, 193, 154], [250, 241, 216], [227, 216, 173], [163, 142, 78], [145, 127, 60], [143, 129, 62], [140, 123, 55], [136, 116, 46], [121, 95, 30], [114, 82, 40]], [[147, 111, 76], [134, 88, 47], [140, 99, 61], [148, 103, 66], [135, 89, 60], [100, 64, 38], [108, 73, 43], [144, 104, 66], [210, 181, 140], [248, 243, 212], [175, 147, 115], [175, 119, 73], [220, 176, 129], [226, 197, 164], [230, 207, 179], [233, 218, 196], [224, 212, 195], [201, 186, 166], [184, 163, 138], [181, 158, 128], [190, 171, 136], [170, 157, 105], [179, 167, 105], [231, 218, 181], [223, 206, 161], [162, 133, 71], [146, 116, 43], [140, 115, 34], [139, 116, 33], [145, 123, 38], [142, 119, 35], [128, 102, 41]], [[152, 114, 80], [117, 75, 37], [114, 80, 48], [123, 90, 57], [126, 91, 56], [122, 83, 48], [93, 58, 32], [179, 154, 138], [238, 226, 212], [248, 243, 229], [170, 134, 104], [185, 132, 88], [241, 214, 177], [230, 218, 195], [187, 169, 142], [180, 160, 131], [166, 146, 115], [146, 119, 85], [149, 116, 79], [157, 124, 83], [184, 157, 110], [216, 195, 141], [212, 198, 152], [236, 221, 197], [236, 212, 176], [166, 125, 63], [136, 85, 16], [134, 81, 13], [130, 83, 13], [127, 86, 16], [137, 105, 27], [151, 128, 54]], [[145, 105, 72], [127, 82, 41], [128, 90, 51], [133, 92, 53], [132, 89, 49], [135, 95, 51], [171, 145, 110], [237, 227, 205], [252, 247, 235], [229, 213, 194], [173, 136, 100], [169, 121, 73], [220, 182, 138], [194, 169, 135], [123, 89, 55], [135, 98, 60], [127, 91, 48], [151, 114, 63], [165, 127, 74], [132, 99, 50], [151, 126, 79], [202, 183, 142], [240, 228, 203], [240, 225, 210], [222, 196, 169], [156, 117, 64], [119, 76, 12], [120, 75, 16], [112, 66, 14], [100, 65, 15], [99, 74, 19], [140, 121, 54]], [[143, 104, 66], [127, 80, 38], [129, 86, 49], [129, 85, 46], [130, 86, 45], [140, 102, 59], [219, 196, 161], [244, 232, 210], [210, 199, 186], [193, 173, 151], [166, 129, 92], [153, 104, 55], [191, 146, 96], [179, 145, 105], [128, 86, 47], [147, 102, 58], [149, 106, 59], [172, 131, 78], [147, 108, 54], [128, 94, 45], [141, 113, 67], [173, 150, 112], [202, 183, 160], [190, 171, 147], [198, 175, 146], [152, 124, 86], [100, 72, 26], [109, 81, 34], [119, 88, 43], [121, 92, 50], [108, 82, 36], [136, 119, 50]], [[143, 104, 64], [125, 76, 32], [131, 85, 48], [128, 81, 43], [123, 81, 39], [153, 117, 76], [148, 118, 85], [166, 141, 118], [188, 166, 147], [182, 156, 132], [171, 134, 99], [165, 115, 69], [195, 148, 99], [190, 153, 110], [152, 108, 66], [143, 95, 49], [152, 105, 56], [153, 110, 58], [142, 102, 51], [141, 102, 54], [135, 101, 55], [136, 101, 60], [148, 110, 74], [141, 106, 65], [141, 111, 68], [138, 113, 71], [111, 100, 37], [111, 111, 31], [121, 118, 35], [129, 116, 39], [138, 116, 45], [179, 162, 83]], [[141, 102, 65], [131, 80, 35], [139, 89, 46], [139, 87, 44], [138, 90, 50], [151, 111, 71], [128, 91, 52], [136, 97, 61], [175, 136, 104], [173, 136, 107], [189, 151, 118], [205, 160, 120], [201, 157, 113], [168, 131, 89], [151, 108, 65], [145, 97, 53], [146, 101, 52], [149, 106, 57], [153, 110, 61], [149, 108, 61], [144, 104, 59], [144, 105, 59], [145, 104, 59], [143, 102, 60], [129, 96, 48], [123, 103, 39], [124, 126, 30], [113, 135, 14], [108, 133, 8], [113, 122, 10], [148, 136, 44], [199, 184, 102]], [[143, 103, 72], [139, 87, 44], [138, 89, 42], [149, 96, 52], [160, 109, 72], [150, 106, 64], [147, 104, 58], [151, 104, 57], [169, 121, 81], [167, 123, 87], [179, 141, 105], [212, 174, 138], [203, 168, 132], [207, 177, 141], [149, 112, 74], [139, 96, 55], [144, 102, 56], [137, 94, 47], [151, 107, 61], [155, 111, 65], [152, 109, 63], [140, 101, 55], [107, 76, 38], [91, 60, 34], [84, 61, 23], [105, 99, 25], [132, 142, 34], [118, 141, 20], [96, 121, 4], [102, 113, 9], [159, 149, 63], [190, 174, 99]], [[149, 107, 74], [133, 80, 37], [136, 88, 48], [147, 99, 59], [150, 104, 63], [153, 109, 62], [157, 112, 67], [162, 117, 78], [175, 131, 96], [190, 145, 107], [166, 124, 84], [202, 168, 133], [224, 197, 168], [197, 175, 148], [192, 165, 133], [180, 144, 107], [146, 106, 64], [126, 82, 40], [141, 97, 52], [156, 112, 66], [153, 109, 61], [115, 74, 29], [77, 43, 14], [79, 50, 25], [93, 73, 33], [126, 117, 47], [133, 134, 39], [119, 116, 31], [113, 99, 24], [140, 121, 42], [187, 165, 91], [154, 132, 75]], [[172, 128, 76], [144, 88, 18], [135, 85, 35], [136, 88, 48], [135, 90, 45], [139, 94, 49], [153, 108, 68], [163, 117, 83], [166, 120, 82], [184, 136, 100], [166, 118, 88], [150, 110, 75], [184, 149, 110], [156, 121, 86], [158, 123, 87], [168, 130, 92], [149, 109, 71], [135, 91, 51], [130, 85, 43], [132, 88, 43], [128, 84, 37], [127, 83, 36], [135, 92, 52], [143, 105, 66], [139, 108, 62], [136, 112, 52], [127, 105, 39], [121, 92, 39], [135, 102, 44], [189, 159, 87], [211, 181, 114], [136, 107, 58]], [[202, 157, 82], [187, 129, 26], [151, 100, 25], [128, 79, 34], [122, 76, 41], [134, 88, 49], [142, 98, 53], [150, 106, 56], [153, 106, 58], [148, 99, 63], [135, 87, 59], [127, 82, 44], [153, 109, 60], [166, 121, 77], [143, 99, 59], [130, 88, 51], [128, 87, 52], [151, 108, 70], [152, 106, 65], [135, 90, 48], [139, 95, 50], [155, 110, 63], [161, 113, 65], [154, 107, 63], [154, 112, 67], [143, 105, 54], [130, 93, 44], [132, 90, 46], [171, 131, 70], [215, 183, 106], [186, 155, 91], [117, 86, 48]], [[216, 174, 87], [193, 136, 16], [168, 122, 19], [151, 111, 35], [131, 88, 34], [126, 82, 35], [138, 94, 49], [144, 100, 53], [142, 95, 53], [137, 92, 51], [120, 78, 34], [131, 87, 41], [145, 99, 52], [144, 101, 57], [137, 94, 54], [127, 83, 48], [126, 82, 51], [139, 94, 60], [153, 108, 69], [149, 104, 63], [140, 95, 53], [135, 91, 46], [147, 103, 57], [148, 105, 59], [149, 108, 62], [149, 109, 63], [137, 101, 54], [143, 107, 57], [203, 167, 102], [206, 173, 105], [124, 93, 49], [71, 48, 26]], [[220, 182, 91], [201, 150, 22], [186, 148, 24], [172, 139, 28], [156, 120, 26], [142, 103, 30], [142, 100, 51], [153, 108, 75], [150, 105, 73], [139, 98, 57], [126, 88, 38], [136, 92, 47], [148, 102, 62], [141, 101, 60], [131, 89, 51], [126, 82, 49], [127, 81, 51], [138, 88, 52], [150, 100, 60], [154, 104, 65], [149, 101, 60], [124, 78, 36], [126, 85, 41], [141, 101, 55], [145, 107, 61], [147, 112, 68], [127, 101, 59], [114, 87, 46], [186, 155, 98], [173, 144, 87], [56, 29, 9], [33, 19, 9]], [[208, 170, 96], [201, 153, 34], [198, 161, 26], [191, 157, 27], [183, 146, 34], [171, 135, 32], [159, 121, 42], [147, 107, 52], [135, 95, 49], [130, 87, 46], [139, 93, 57], [147, 98, 62], [144, 95, 55], [145, 99, 57], [137, 91, 51], [136, 89, 52], [137, 90, 54], [148, 102, 58], [152, 106, 60], [150, 103, 61], [155, 110, 64], [138, 94, 46], [120, 76, 33], [128, 84, 39], [142, 102, 58], [135, 103, 62], [90, 69, 40], [50, 24, 11], [137, 105, 60], [160, 133, 70], [56, 31, 7], [53, 34, 20]], [[180, 139, 96], [173, 123, 42], [186, 144, 30], [194, 153, 25], [198, 158, 34], [201, 164, 36], [189, 153, 32], [173, 137, 32], [156, 118, 38], [139, 99, 38], [142, 97, 49], [145, 97, 56], [141, 92, 52], [141, 93, 52], [139, 91, 51], [140, 91, 53], [143, 95, 58], [139, 99, 60], [138, 98, 60], [143, 96, 56], [146, 93, 43], [135, 84, 33], [117, 80, 38], [112, 72, 29], [122, 81, 39], [104, 67, 30], [58, 31, 11], [34, 5, 0], [131, 94, 57], [184, 148, 94], [97, 62, 34], [83, 53, 34]], [[177, 144, 116], [168, 129, 94], [179, 142, 87], [188, 149, 67], [202, 168, 68], [218, 189, 76], [218, 191, 72], [207, 181, 70], [191, 163, 79], [175, 143, 82], [166, 132, 86], [163, 128, 92], [163, 127, 94], [161, 123, 92], [153, 114, 84], [159, 120, 90], [162, 124, 93], [149, 116, 91], [140, 104, 83], [148, 103, 77], [161, 105, 69], [144, 95, 55], [112, 90, 59], [119, 91, 58], [130, 96, 65], [120, 87, 59], [92, 67, 46], [103, 78, 57], [170, 140, 104], [216, 184, 140], [151, 118, 84], [123, 92, 72]]]
    x_train = np.array([image]*5000)
    y_train = np.array([[6]]*5000)
    x_test = np.array([image]*1000)
    y_test = np.array([[6]]*1000)
    return (x_train, y_train), (x_test, y_test)

