#!/usr/bin/env python3
"""
Check if all required dependencies are installed.
"""

def check_dependencies():
    """Check if all required dependencies are available."""
    
    print("🔍 Checking Dependencies")
    print("=" * 30)
    
    required_packages = [
        ('pandas', 'pandas'),
        ('numpy', 'numpy'), 
        ('requests', 'requests'),
        ('yfinance', 'yfinance'),
        ('pyyaml', 'yaml'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('plotly', 'plotly'),
        ('scipy', 'scipy'),
        ('scikit-learn', 'sklearn'),
        ('statsmodels', 'statsmodels'),
        ('arch', 'arch')
    ]
    
    missing_packages = []
    installed_packages = []
    
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
            installed_packages.append(package_name)
            print(f"✅ {package_name}")
        except ImportError:
            missing_packages.append(package_name)
            print(f"❌ {package_name}")
    
    print(f"\n📊 Summary:")
    print(f"   Installed: {len(installed_packages)}/{len(required_packages)}")
    print(f"   Missing: {len(missing_packages)}")
    
    if missing_packages:
        print(f"\n🔧 Missing packages: {', '.join(missing_packages)}")
        print(f"\n💡 To install missing packages:")
        print(f"   py -m pip install {' '.join(missing_packages)}")
    else:
        print(f"\n🎉 All dependencies are installed!")
    
    return len(missing_packages) == 0

if __name__ == "__main__":
    success = check_dependencies()
    exit(0 if success else 1)
