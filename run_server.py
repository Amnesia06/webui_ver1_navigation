#!/usr/bin/env python3
"""
Farm Robot Path Planner Web Server
Run this file to start the web application
"""

import os
import sys
from web_app import app

if __name__ == '__main__':
    print("🤖 Farm Robot Path Planner Web Server")
    print("=" * 50)
    print("🚀 Starting server...")
    print("🌐 Open your browser and go to: http://localhost:5000")
    print("📱 Or from another device: http://[your-ip]:5000")
    print("⏹️  Press Ctrl+C to stop the server_")
    print("=" * 50)
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        sys.exit(1)
