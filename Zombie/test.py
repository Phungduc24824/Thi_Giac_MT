import torch
import torch.nn as nn

class SimplePhysNet(nn.Module):
    def __init__(self):
        super(SimplePhysNet, self).__init__()
        
        # Lớp tích chập 3D số 1: Trích xuất đặc trưng không gian - thời gian ban đầu
        # Input: (Batch, 3, Frames, H, W) -> Ví dụ: (1, 3, 64, 64, 64)
        self.conv1 = nn.Conv3d(in_channels=3, out_channels=16, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(16)
        self.relu1 = nn.ReLU()
        
        # Lớp Pooling 3D: Giảm kích thước không gian nhưng giữ nguyên chiều thời gian
        # Chúng ta chỉ muốn nén ảnh nhỏ lại, chứ không muốn mất đi tần số nhịp tim theo thời gian
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)) # H và W giảm 1 nửa
        
        # Lớp tích chập 3D số 2
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        # Lớp tích chập toàn cục để nén không gian (H, W) về 1x1, chỉ giữ lại tín hiệu 1D theo Thời gian
        self.global_conv = nn.Conv3d(in_channels=32, out_channels=1, kernel_size=(1, 16, 16), stride=1)
        
    def forward(self, x):
        # x có hình dạng: [Batch, Channels, Frames, Height, Width]
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        # Giả sử sau 2 lần pool, H và W từ 64 giảm xuống còn 16
        x = self.global_conv(x) # Đầu ra lúc này: [Batch, 1, Frames, 1, 1]
        
        # Loại bỏ các chiều kích thước thừa bằng hàm squeeze
        rppg_signal = x.squeeze(1).squeeze(-1).squeeze(-1) # Đầu ra cuối cùng: [Batch, Frames]
        
        return rppg_signal

# --- CHẠY THỬ MÔ HÌNH VỚI DỮ LIỆU GIẢ LẬP ---
if __name__ == "__main__":
    model = SimplePhysNet()
    
    # Giả lập 1 kiến trúc input: 1 video, 3 kênh màu, dài 64 khung hình, ảnh kích thước 64x64
    fake_video_tensor = torch.randn(1, 3, 64, 64, 64) 
    
    output_signal = model(fake_video_tensor)
    print("Kích thước đầu ra của tín hiệu rPPG dự đoán:", output_signal.shape) 
    # Kết quả sẽ là [1, 64] -> Tương ứng với chuỗi tín hiệu mạch đập dài 64 điểm theo thời gian.