import numpy as np
import cv2 as cv
import os

class Seam:
    def __init__(self, input_image_path: str, output_image_height: int, output_image_width: int, edge_detect_method: str = 'roberts') -> None:
        '''
        input_image_path    要处理的图像路径\n
        output_image_height 要求的图像高度\n
        output_image_width  要求的图像宽度\n
        edge_detect_method  要求使用的边缘检测方法，默认使用Roberts算子\n
        '''

        # 初始化参数
        self.input_image_path = input_image_path
        self.output_image_height = output_image_height
        self.output_image_width = output_image_width
        self.edge_detect_method = edge_detect_method

        # 读取图像
        self.input_image = cv.imread(input_image_path)
        self.input_image_height, self.input_image_width = self.input_image.shape[: 2]
        self.output_image = np.copy(self.input_image)

        self.start()

    def start(self) -> None:
        dHeight = self.output_image_height - self.input_image_height
        dWidth = self.output_image_width - self.input_image_width
        self.total_operation_times = abs(dHeight) + abs(dWidth)
        self.count = 0

        print('In horizontal direction, it needs to be operated {0} times.'.format(
            abs(dWidth)))
        print('In vertical direction, it needs to be operated {0} times.'.format(
            abs(dHeight)))
        print('To sum up, it needs to be operated {0} times in total.'.format(
            self.total_operation_times))


        self.show_progress()

        # 水平方向
        if dWidth < 0:
            self.seam_remove(abs(dWidth))
        else:
            self.seam_insert(dWidth)


        # 垂直方向
        if dHeight < 0:
            self.image_rotate(1)
            self.seam_remove(abs(dHeight))
            self.image_rotate(-1)
        else:
            self.image_rotate(1)
            self.seam_insert(dHeight)
            self.image_rotate(-1)
        
        self.save_image()

    def show_progress(self) -> None:
        '''
        显示工作进度\n
        '''
        done = self.count
        total = self.total_operation_times
        length = 50
        ratio = 1.0*done/total
        doneBar = round(ratio*length)
        undoneBar = length - doneBar
        print(f"[{'>'*doneBar}{'-'*undoneBar}] ({done} / {total})", end='\r')
        if ratio == 1.0:
            print()

    def image_rotate(self, method: int) -> None:
        '''
        若method为1，则向右旋转self.output_image 90度\n
        若method为-1，则向左旋转self.output_image 90度\n
        '''
        if method == 1:
            self.output_image = cv.rotate(
                self.output_image, cv.ROTATE_90_CLOCKWISE)
        elif method == -1:
            self.output_image = cv.rotate(
                self.output_image, cv.ROTATE_90_COUNTERCLOCKWISE)

    def seam_remove(self, total: int) -> None:
        '''
        删除total数量的seam\n
        '''
        for i in range(total):
            self.calculate_energy_map()
            self.calculate_cost_map()
            self.find_seam()
            self.delete_seam()
            self.count += 1
            self.show_progress()

    def seam_insert(self, total: int) -> None:
        '''
        插入total数量的seam\n
        '''
        tmp_image = np.copy(self.output_image)
        seam_list = []

        for i in range(total):
            self.calculate_energy_map()
            self.calculate_cost_map()
            self.find_seam()
            self.delete_seam()
            seam_list.append(self.best_path)

        self.output_image = np.copy(tmp_image)

        for i in range(total):
            selected_seam = seam_list.pop()
            self.insert_seam(selected_seam)
            for seam in seam_list:
                seam[np.where(seam >= selected_seam)] += 1
        
            self.count += 1
            self.show_progress()

    def sobel(self, src: np.ndarray) -> np.ndarray:
        '''
        使用Sobel算子计算能量函数\n
        '''
        output = np.abs(cv.Sobel(src, -1, 1, 0)) + \
            np.abs(cv.Sobel(src, -1, 0, 1))
        return output

    def prewitt(self, src: np.ndarray) -> np.ndarray:
        '''
        使用Prewitt算子计算能量函数\n
        '''
        kernel_x = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
        kernel_y = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
        x = cv.filter2D(src, -1, kernel_x)
        y = cv.filter2D(src, -1, kernel_y)
        output = np.abs(x) + np.abs(y)
        return output

    def roberts(self, src: np.ndarray) -> np.ndarray:
        '''
        使用Roberts算子计算能量函数\n
        '''
        kernel_x = np.array([[-1, 0], [0, 1]], dtype=int)
        kernel_y = np.array([[0, -1], [1, 0]], dtype=int)
        x = cv.filter2D(src, -1, kernel_x)
        y = cv.filter2D(src, -1, kernel_y)
        output = np.abs(x) + np.abs(y)
        return output

    def canny(self, src: np.ndarray) -> np.ndarray:
        '''
        使用Canny边缘检测器计算能量函数\n
        '''
        output = np.abs(cv.Canny(src, 100, 200))
        return output

    def calculate_energy_map(self) -> None:
        '''
        计算能量图\n
        将原图像拆分成b,g,r三个通道，分别在三个通道上使用Scharr算子
        计算x方向和y方向上的差分，取绝对值再求和就是最后的能量图\n
        '''
        b, g, r = cv.split(self.output_image)
        if (self.edge_detect_method == 'sobel'):
            b_energy = self.sobel(b)
            g_energy = self.sobel(g)
            r_energy = self.sobel(r)
        elif (self.edge_detect_method == 'prewitt'):
            b_energy = self.prewitt(b)
            g_energy = self.prewitt(g)
            r_energy = self.prewitt(r)
        elif (self.edge_detect_method == 'canny'):
            b_energy = self.canny(b)
            g_energy = self.canny(g)
            r_energy = self.canny(r)
        elif (self.edge_detect_method == 'roberts'):
            b_energy = self.roberts(b)
            g_energy = self.roberts(g)
            r_energy = self.roberts(r)
        self.energy_map = b_energy + g_energy + r_energy
        cv.imshow('energy_map', self.energy_map)
        cv.imwrite('./output/energy_map.png', self.energy_map)
        cv.waitKey(0)

    def calculate_cost_map(self) -> None:
        '''
        计算代价图\n
        '''
        m, n = self.energy_map.shape
        cost_map = np.copy(self.energy_map).astype(int)
        path_map_up = np.zeros([n, m], dtype=int)
        path_map_down = np.zeros([n,m], dtype=int)
        for j in range(n):
            path_map_up[j,0] = j
        for i in range(1, m):
            for j in range(n):
                if (j == 0):
                    min_index = np.argmin(cost_map[i-1][j:j+1])+j
                elif (j == n-1):
                    min_index = np.argmin(cost_map[i-1][j-1:j])+j-1
                else:
                    min_index = np.argmin(cost_map[i-1][j-1:j+1])+j-1
                cost_map[i][j] += cost_map[i-1][min_index]
                path_map_down[j,:] = path_map_up[min_index,:]
                path_map_down[j,i] = j
            path_map_up = np.copy(path_map_down)
        self.cost_map = cost_map
        self.path_map = path_map_down

    def find_seam(self) -> int:
        '''
        找到最优路径\n
        '''
        mincost_index = np.argmin(self.cost_map[-1][:])
        self.best_path = self.path_map[mincost_index,:]

        tmp_image = self.output_image
        m = tmp_image.shape[0]
        path = self.best_path
        for i in range(m):
            tmp_image[i, path[i], 0] = tmp_image[i,
                                                 path[i], 1] = tmp_image[i, path[i], 2] = 255
        cv.waitKey(100)

    def delete_seam(self) -> None:
        '''
        删除指定编号的缝\n
        '''
        path = self.best_path
        m, n = self.output_image.shape[0:2]
        tmp_image = np.zeros((m, n-1, 3)).astype('uint8')
        for i in range(m):
            tmp_image[i, :path[i]] = self.output_image[i, :path[i]]
            tmp_image[i, path[i]:] = self.output_image[i, path[i]+1:]
        self.output_image = np.copy(tmp_image)

    def insert_seam(self, path) -> None:
        '''
        插入指定的缝\n
        '''
        m, n = self.output_image.shape[0:2]
        tmp_image = np.zeros((m, n+1, 3)).astype('uint8')
        for i in range(m):
            j = path[i]
            for ch in range(3):
                if j == 0:
                    avg = np.average(self.output_image[i, j:j+2, ch])
                    tmp_image[i, j, ch] = self.output_image[i, j, ch]
                    tmp_image[i, j+1, ch] = avg
                    tmp_image[i, j+1:, ch] = self.output_image[i, j:, ch]
                else:
                    avg = np.average(self.output_image[i, j-1:j+1, ch])
                    tmp_image[i, :j, ch] = self.output_image[i, :j, ch]
                    tmp_image[i, j, ch] = avg
                    tmp_image[i, j+1:, ch] = self.output_image[i, j:, ch]
        self.output_image = np.copy(tmp_image)

    def save_image(self) -> None:
        '''
        保存输出的图片\n
        '''
        dir = './output'
        if not os.path.exists(dir):
            os.makedirs(dir)
        cnt = len(os.listdir(dir))
        cv.imwrite('./output/{0}.png'.format(cnt), self.output_image)