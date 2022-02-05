from Seam import Seam
import os

def main() -> None:
    obj = Seam('image/test.png', 280, 280, edge_detect_method='prewitt')

if __name__ == '__main__':
    main()