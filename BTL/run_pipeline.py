"""
Quick start guide cho Optimized Realtime Pipeline
Chạy pipeline với metrics tracking
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from web_cam_optimized import OptimizedRealtimePipeline


def main():
    print("\n" + "="*70)
    print("🚀 OPTIMIZED REALTIME rPPG PIPELINE (Quick Start)")
    print("="*70 + "\n")
    
    print("🎯 Chọn mode chạy:\n")
    print("  1. Auto-select model (khuyên dùng) ⚡")
    print("  2. Mobile model (nhanh nhất, 24 FPS) 📱")
    print("  3. Efficient model (cân bằng, 20 FPS) ⚖️")
    print("  4. Advanced model (chính xác nhất) 🎯")
    print("  5. Export models only ⬇️")
    print()
    
    choice = input("Chọn (1-5): ").strip()
    
    model_map = {
        '1': (None, True),   # (model, auto_model)
        '2': ('mobile', False),
        '3': ('efficient', False),
        '4': ('advanced', False),
    }
    
    if choice == '5':
        print("\n[INFO] Exporting models...\n")
        pipeline = OptimizedRealtimePipeline()
        pipeline.export_optimized_model()
        print("\n✅ Models exported to ket_qua/\n")
        return
    
    if choice not in model_map:
        print("[ERROR] Invalid choice")
        return
    
    model, auto_model = model_map[choice]
    
    print("\n📋 Chọn chế độ:\n")
    print("  1. Realtime display (hiển thị webcam)")
    print("  2. Benchmark mode (không hiển thị, đo FPS)")
    print()
    
    mode = input("Chọn (1-2): ").strip()
    
    show_display = mode != '2'
    
    print("\n" + "-"*70)
    print("🔄 Khởi động pipeline...")
    print("-"*70 + "\n")
    
    try:
        pipeline = OptimizedRealtimePipeline(
            model_type=model,
            window_size=24,
            auto_model=auto_model
        )
        
        pipeline.run(show_display=show_display)
        
    except KeyboardInterrupt:
        print("\n\n[INFO] Pipeline stopped by user")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
