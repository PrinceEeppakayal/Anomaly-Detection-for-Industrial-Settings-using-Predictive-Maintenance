"""
📧 EMAIL NOTIFICATION SYSTEM FOR ANOMALY DETECTION
==================================================

Sends email alerts based on anomaly severity:
- CRITICAL: Immediate notification (multiple anomalies detected)
- WARNING: Alert notification (anomalies detected, schedule maintenance)
- INFO: Informational notification (minor anomaly, monitor system)

Supports:
- Gmail SMTP
- Outlook/Office365 SMTP
- Custom SMTP servers
- HTML formatted emails with color-coded severity
- Attachment support for logs/screenshots

Author: Industrial IoT Monitoring System
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime
import os
import json

class EmailNotificationSystem:
    """Email notification system for anomaly alerts"""
    
    def __init__(self, config_file="config/email_config.json"):
        """
        Initialize email notifier with configuration
        
        Args:
            config_file: Path to JSON configuration file
        """
        self.config_file = config_file
        self.config = self.load_config()
        
        # Notification cooldown to prevent spam
        self.last_notification = {
            'CRITICAL': None,
            'WARNING': None,
            'INFO': None
        }
        
        # Load cooldown periods from config or use defaults
        config_cooldown = self.config.get('notification_cooldown', {})
        self.cooldown_periods = {
            'CRITICAL': config_cooldown.get('CRITICAL', 300),    # Default 5 minutes
            'WARNING': config_cooldown.get('WARNING', 900),      # Default 15 minutes
            'INFO': config_cooldown.get('INFO', 1800)            # Default 30 minutes
        }
        
        print(f"📧 Email cooldown periods loaded: CRITICAL={self.cooldown_periods['CRITICAL']}s, "
              f"WARNING={self.cooldown_periods['WARNING']}s, INFO={self.cooldown_periods['INFO']}s")
        
        self.notification_count = 0
        
    def load_config(self):
        """Load email configuration from file or return defaults"""
        
        # Default configuration template
        default_config = {
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 587,
            "sender_email": "your-email@gmail.com",
            "sender_password": "your-app-password",
            "recipient_emails": ["recipient1@example.com", "recipient2@example.com"],
            "enable_notifications": True,
            "notification_cooldown": {
                "CRITICAL": 300,
                "WARNING": 900,
                "INFO": 1800
            }
        }
        
        # Try to load from file
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                    print(f"✅ Email configuration loaded from {self.config_file}")
                    return loaded_config
            except Exception as e:
                print(f"⚠️  Could not load config file: {e}")
                print("   Using default configuration")
        else:
            # Create config directory if it doesn't exist
            config_dir = os.path.dirname(self.config_file)
            if config_dir and not os.path.exists(config_dir):
                os.makedirs(config_dir, exist_ok=True)
            
            # Save default configuration for user to edit
            try:
                with open(self.config_file, 'w') as f:
                    json.dump(default_config, f, indent=4)
                print(f"📝 Created default config file: {self.config_file}")
                print(f"   Please edit this file with your email credentials!")
            except Exception as e:
                print(f"⚠️  Could not create config file: {e}")
        
        return default_config
    
    def create_html_email(self, severity, station_id, timestamp, anomaly_data, 
                          maintenance_message, model_stats):
        """
        Create HTML formatted email with color-coded severity
        
        Args:
            severity: CRITICAL, WARNING, or INFO
            station_id: Base station identifier
            timestamp: Current timestamp
            anomaly_data: Dictionary with sensor readings
            maintenance_message: Maintenance recommendation
            model_stats: AI model performance statistics
            
        Returns:
            HTML string for email body
        """
        
        # Color scheme based on severity
        colors = {
            'CRITICAL': {'bg': '#ff4444', 'text': '#ffffff', 'border': '#cc0000'},
            'WARNING': {'bg': '#ffaa00', 'text': '#000000', 'border': '#cc8800'},
            'INFO': {'bg': '#4488ff', 'text': '#ffffff', 'border': '#2266cc'}
        }
        
        severity_color = colors.get(severity, colors['INFO'])
        
        html = f"""
        <html>
        <head>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    background-color: #f4f4f4;
                    margin: 0;
                    padding: 20px;
                }}
                .container {{
                    background-color: #ffffff;
                    border-radius: 10px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    max-width: 600px;
                    margin: 0 auto;
                    overflow: hidden;
                }}
                .header {{
                    background-color: {severity_color['bg']};
                    color: {severity_color['text']};
                    padding: 20px;
                    text-align: center;
                    border-bottom: 5px solid {severity_color['border']};
                }}
                .header h1 {{
                    margin: 0;
                    font-size: 28px;
                }}
                .severity-badge {{
                    background-color: rgba(255,255,255,0.3);
                    padding: 5px 15px;
                    border-radius: 20px;
                    display: inline-block;
                    margin-top: 10px;
                    font-weight: bold;
                }}
                .content {{
                    padding: 30px;
                }}
                .info-section {{
                    background-color: #f9f9f9;
                    border-left: 4px solid {severity_color['bg']};
                    padding: 15px;
                    margin: 15px 0;
                }}
                .info-section h3 {{
                    margin-top: 0;
                    color: {severity_color['border']};
                }}
                .sensor-grid {{
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 10px;
                    margin: 15px 0;
                }}
                .sensor-item {{
                    background-color: #ffffff;
                    border: 1px solid #ddd;
                    padding: 10px;
                    border-radius: 5px;
                }}
                .sensor-label {{
                    font-size: 12px;
                    color: #666;
                    margin-bottom: 5px;
                }}
                .sensor-value {{
                    font-size: 20px;
                    font-weight: bold;
                    color: #333;
                }}
                .stats-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 10px 0;
                }}
                .stats-table th {{
                    background-color: #f0f0f0;
                    padding: 8px;
                    text-align: left;
                    border-bottom: 2px solid #ddd;
                }}
                .stats-table td {{
                    padding: 8px;
                    border-bottom: 1px solid #eee;
                }}
                .footer {{
                    background-color: #f0f0f0;
                    padding: 15px;
                    text-align: center;
                    font-size: 12px;
                    color: #666;
                }}
                .alert-icon {{
                    font-size: 48px;
                    margin-bottom: 10px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <div class="alert-icon">{"🚨" if severity == "CRITICAL" else "⚠️" if severity == "WARNING" else "ℹ️"}</div>
                    <h1>Anomaly Detection Alert</h1>
                    <div class="severity-badge">{severity}</div>
                </div>
                
                <div class="content">
                    <div class="info-section">
                        <h3>🗼 Station Information</h3>
                        <p><strong>Station ID:</strong> {station_id}</p>
                        <p><strong>Timestamp:</strong> {timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
                        <p><strong>Anomaly Type:</strong> {anomaly_data.get('anomaly_type', 'Unknown')}</p>
                    </div>
                    
                    <div class="info-section">
                        <h3>📊 Current Sensor Readings</h3>
                        <div class="sensor-grid">
                            <div class="sensor-item">
                                <div class="sensor-label">🌡️ Temperature</div>
                                <div class="sensor-value">{anomaly_data.get('temperature_C', 0):.1f}°C</div>
                            </div>
                            <div class="sensor-item">
                                <div class="sensor-label">⚡ Power</div>
                                <div class="sensor-value">{anomaly_data.get('power_consumption_W', 0):.0f}W</div>
                            </div>
                            <div class="sensor-item">
                                <div class="sensor-label">📡 Signal</div>
                                <div class="sensor-value">{anomaly_data.get('signal_strength_dBm', 0):.1f}dBm</div>
                            </div>
                            <div class="sensor-item">
                                <div class="sensor-label">🌐 Network Load</div>
                                <div class="sensor-value">{anomaly_data.get('network_load_percent', 0):.1f}%</div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="info-section">
                        <h3>🔧 Maintenance Recommendation</h3>
                        <p><strong>{maintenance_message}</strong></p>
                    </div>
                    
                    <div class="info-section">
                        <h3>🤖 AI Model Performance</h3>
                        <table class="stats-table">
                            <tr>
                                <th>Metric</th>
                                <th>Value</th>
                            </tr>
                            <tr>
                                <td>Model Version</td>
                                <td>v{model_stats.get('version', 0)}</td>
                            </tr>
                            <tr>
                                <td>Accuracy</td>
                                <td>{model_stats.get('accuracy', 0):.1f}%</td>
                            </tr>
                            <tr>
                                <td>Precision</td>
                                <td>{model_stats.get('precision', 0):.1f}%</td>
                            </tr>
                            <tr>
                                <td>Recall</td>
                                <td>{model_stats.get('recall', 0):.1f}%</td>
                            </tr>
                            <tr>
                                <td>Total Predictions</td>
                                <td>{model_stats.get('total_predictions', 0):,}</td>
                            </tr>
                        </table>
                    </div>
                </div>
                
                <div class="footer">
                    <p>🗼 Industrial IoT Anomaly Detection System v2.0</p>
                    <p>This is an automated notification. Please do not reply to this email.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def should_send_notification(self, severity):
        """
        Check if enough time has passed since last notification of this severity
        (Prevents email spam)
        
        Args:
            severity: CRITICAL, WARNING, or INFO
            
        Returns:
            Boolean indicating if notification should be sent
        """
        if not self.config.get('enable_notifications', True):
            print(f"⚠️  Email notifications are DISABLED in config")
            return False
        
        last_time = self.last_notification.get(severity)
        if last_time is None:
            return True
        
        cooldown = self.cooldown_periods.get(severity, 300)
        time_since_last = (datetime.now() - last_time).total_seconds()
        
        if time_since_last < cooldown:
            time_remaining = int(cooldown - time_since_last)
            print(f"⏳ Email [{severity}] blocked by cooldown. Wait {time_remaining}s more.")
            return False
        
        return True
    
    def send_email_notification(self, severity, station_id, timestamp, 
                                anomaly_data, maintenance_message, model_stats,
                                attachments=None):
        """
        Send email notification about detected anomaly
        
        Args:
            severity: CRITICAL, WARNING, or INFO
            station_id: Base station identifier
            timestamp: Current timestamp
            anomaly_data: Dictionary with sensor readings
            maintenance_message: Maintenance recommendation
            model_stats: AI model performance statistics
            attachments: List of file paths to attach (optional)
            
        Returns:
            Boolean indicating success
        """
        
        # Check if notification should be sent (cooldown check)
        if not self.should_send_notification(severity):
            return False
        
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"[{severity}] Anomaly Alert - {station_id} - {timestamp.strftime('%H:%M:%S')}"
            msg['From'] = self.config['sender_email']
            msg['To'] = ', '.join(self.config['recipient_emails'])
            
            # Create HTML content
            html_content = self.create_html_email(
                severity, station_id, timestamp, anomaly_data,
                maintenance_message, model_stats
            )
            
            # Attach HTML content
            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)
            
            # Add attachments if provided
            if attachments:
                for file_path in attachments:
                    if os.path.exists(file_path):
                        with open(file_path, 'rb') as f:
                            part = MIMEBase('application', 'octet-stream')
                            part.set_payload(f.read())
                            encoders.encode_base64(part)
                            part.add_header(
                                'Content-Disposition',
                                f'attachment; filename={os.path.basename(file_path)}'
                            )
                            msg.attach(part)
            
            # Connect to SMTP server and send email
            with smtplib.SMTP(self.config['smtp_server'], self.config['smtp_port']) as server:
                server.starttls()  # Enable TLS encryption
                server.login(self.config['sender_email'], self.config['sender_password'])
                server.send_message(msg)
            
            # Update last notification time
            self.last_notification[severity] = datetime.now()
            self.notification_count += 1
            
            print(f"📧 Email notification sent! [{severity}] to {len(self.config['recipient_emails'])} recipient(s)")
            return True
            
        except Exception as e:
            print(f"❌ Failed to send email notification: {e}")
            return False
    
    def send_test_email(self):
        """Send a test email to verify configuration"""
        
        test_data = {
            'anomaly_type': 'Test Anomaly',
            'temperature_C': 75.5,
            'power_consumption_W': 5500,
            'signal_strength_dBm': -85.3,
            'network_load_percent': 92.1
        }
        
        test_stats = {
            'version': 1,
            'accuracy': 98.5,
            'precision': 95.2,
            'recall': 91.8,
            'total_predictions': 1500
        }
        
        print("📧 Sending test email...")
        success = self.send_email_notification(
            severity='INFO',
            station_id='TEST-STATION',
            timestamp=datetime.now(),
            anomaly_data=test_data,
            maintenance_message='This is a test notification. If you receive this, email notifications are working correctly!',
            model_stats=test_stats
        )
        
        if success:
            print("✅ Test email sent successfully!")
        else:
            print("❌ Test email failed. Please check your configuration.")
        
        return success


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("📧 Email Notification System Demo")
    print("=" * 70)
    
    # Initialize notifier
    notifier = EmailNotificationSystem()
    
    print("\n📝 Current Configuration:")
    print(f"   SMTP Server: {notifier.config['smtp_server']}")
    print(f"   Sender: {notifier.config['sender_email']}")
    print(f"   Recipients: {notifier.config['recipient_emails']}")
    print(f"   Notifications Enabled: {notifier.config['enable_notifications']}")
    
    print("\n" + "=" * 70)
    print("⚠️  IMPORTANT: Edit config/email_config.json before using!")
    print("   1. Add your email address")
    print("   2. Add your app password (not your regular password)")
    print("   3. Add recipient email addresses")
    print("\n📖 For Gmail users:")
    print("   - Enable 2-factor authentication")
    print("   - Generate App Password: https://myaccount.google.com/apppasswords")
    print("   - Use the 16-character app password in config")
    print("=" * 70)
    
    # Ask user if they want to send test email
    response = input("\n❓ Send test email? (yes/no): ").strip().lower()
    
    if response in ['yes', 'y']:
        notifier.send_test_email()
    else:
        print("ℹ️  Test email cancelled. Configure your settings and run again.")
