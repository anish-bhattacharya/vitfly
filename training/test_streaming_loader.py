"""Test streaming data loader with full 580-trajectory dataset."""

import sys
import time
import psutil
import os

# Add training directory to path
sys.path.insert(0, '/root/vitfly/training')

from dataloading_streaming import dataloader_streaming

def monitor_memory():
    """Return current memory usage in GB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)

def main():
    print(f'[TEST] Starting streaming data loader test')
    print(f'[TEST] Initial memory: {monitor_memory():.2f} GB')
    
    data_dir = '/root/vitfly/training/datasets/data_full'
    
    start_time = time.time()
    
    try:
        # Load full dataset (580 trajectories)
        result = dataloader_streaming(
            data_dir=data_dir,
            val_split=0.1,
            short=0,  # 0 = load all trajectories
            seed=42,
            chunk_size=50
        )
        
        elapsed = time.time() - start_time
        final_memory = monitor_memory()
        
        # Unpack result
        train_data, val_data, _, _ = result
        train_meta, train_ims, train_lengths, train_vels, train_quats, train_ctbr = train_data
        val_meta, val_ims, val_lengths, val_vels, val_quats, val_ctbr = val_data
        
        print(f'\n[TEST] ✅ SUCCESS - Data loading completed!')
        print(f'[TEST] Time elapsed: {elapsed:.1f}s')
        print(f'[TEST] Final memory: {final_memory:.2f} GB')
        print(f'[TEST] Train samples: {len(train_ims)}')
        print(f'[TEST] Val samples: {len(val_ims)}')
        print(f'[TEST] Train trajectories: {len(train_lengths)}')
        print(f'[TEST] Val trajectories: {len(val_lengths)}')
        print(f'[TEST] Image shape: {train_ims.shape}')
        print(f'[TEST] Metadata shape: {train_meta.shape}')
        
        return True
        
    except Exception as e:
        print(f'\n[TEST] ❌ FAILED - Error during data loading:')
        print(f'[TEST] {type(e).__name__}: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
