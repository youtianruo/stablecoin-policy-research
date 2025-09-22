#!/usr/bin/env python3
"""
Test analysis pipeline configuration.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.config import load_config

def test_config():
    """Test that the analysis config has all required fields."""
    
    print("🔍 Testing Analysis Configuration")
    print("=" * 40)
    
    try:
        # Test main config
        print("\n1️⃣ Testing main config...")
        config = load_config('configs/config.yaml')
        
        if 'output' in config:
            print("✅ Output section found")
            output = config['output']
            
            required_dirs = ['results_dir', 'figures_dir']
            for dir_key in required_dirs:
                if dir_key in output:
                    print(f"✅ {dir_key}: {output[dir_key]}")
                else:
                    print(f"❌ Missing {dir_key}")
        else:
            print("❌ No output section found")
        
        # Test event study config
        print("\n2️⃣ Testing event study config...")
        event_config = load_config('configs/model_event_study.yaml')
        
        if 'output' in event_config:
            print("✅ Event study output section found")
            output = event_config['output']
            
            required_dirs = ['results_dir', 'figures_dir']
            for dir_key in required_dirs:
                if dir_key in output:
                    print(f"✅ {dir_key}: {output[dir_key]}")
                else:
                    print(f"❌ Missing {dir_key}")
        else:
            print("❌ No output section in event study config")
        
        print("\n🎉 Configuration test completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_config()
