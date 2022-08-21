# -
随机利用翻转变换、旋转变换、高斯模糊、曝光改变、增加椒盐噪声 、色彩抖动种某一种方式，对已有数据集进行增强，并对原有json文件进行修改生成对应的json标签文件。

annotation:存放初始图片的json

lwir_ori,visible_ori:存放初始图片

lwir,visible:增强后的图片

tmp_lwir,tmp_visible:临时生成的增强图片

label_json_lwir,label_json_visible:增强后的图片的json

