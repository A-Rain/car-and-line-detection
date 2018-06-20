Imgwidth = 64
Imgheight = 64



block_x = 16
block_y = 16
Window_x = 64
Window_y = 64
block_stride_x = 8
block_stride_y = 8
cell_x = 8
cell_y = 8
Window_stride_x = 8
Window_stride_y = 8
dimension = 9 * (block_x/cell_x) * (block_y/cell_y) * \
                (1+(Window_x-block_x)/block_stride_x) * (1+(Window_y-block_y)/block_stride_y)

print dimension
