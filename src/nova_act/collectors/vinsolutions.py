"""
VinSolutions-specific data collection implementation.
"""

from typing import Dict, Any, List, Optional
from ..core import NovaActManager

class VinSolutionsCollector:
    """Handles data collection from VinSolutions CRM."""
    
    def __init__(self, nova_manager: NovaActManager):
        """
        Initialize the VinSolutions collector.
        
        Args:
            nova_manager: Instance of NovaActManager
        """
        self.nova = nova_manager
        
        # Define VinSolutions-specific selectors and paths
        self.selectors = {
            "login": {
                "username": "#username",
                "password": "#password",
                "submit": "#loginButton"
            },
            "reports": {
                "menu": "#reportsMenu",
                "sales": "#salesReports",
                "leads": "#leadReports",
                "inventory": "#inventoryReports"
            },
            "download": {
                "button": "#downloadReport",
                "format_select": "#exportFormat"
            }
        }
    
    async def collect_sales_report(self, credentials: Dict[str, Any]) -> Dict[str, Any]:
        """Collect sales report data."""
        report_config = {
            "type": "sales",
            "path": ["reports", "sales"],
            "selectors": self.selectors,
            "download_selector": self.selectors["download"]["button"],
            "file_pattern": "*.csv",
            "format": "csv"
        }
        
        return await self.nova.collect_data(
            vendor="vinsolutions",
            credentials=credentials,
            report_config=report_config
        )
    
    async def collect_lead_report(self, credentials: Dict[str, Any]) -> Dict[str, Any]:
        """Collect lead report data."""
        report_config = {
            "type": "leads",
            "path": ["reports", "leads"],
            "selectors": self.selectors,
            "download_selector": self.selectors["download"]["button"],
            "file_pattern": "*.csv",
            "format": "csv"
        }
        
        return await self.nova.collect_data(
            vendor="vinsolutions",
            credentials=credentials,
            report_config=report_config
        )
    
    async def collect_inventory_report(self, credentials: Dict[str, Any]) -> Dict[str, Any]:
        """Collect inventory report data."""
        report_config = {
            "type": "inventory",
            "path": ["reports", "inventory"],
            "selectors": self.selectors,
            "download_selector": self.selectors["download"]["button"],
            "file_pattern": "*.csv",
            "format": "csv"
        }
        
        return await self.nova.collect_data(
            vendor="vinsolutions",
            credentials=credentials,
            report_config=report_config
        )

    def collect_report(self, config: dict) -> Optional[str]:
        """Collects the specified report from VinSolutions."""
        print("[INFO][VinSolutions] Starting report collection...")
        if not self._login(config.get('username'), config.get('password')):
            return None

        report_path = config.get('report_path')
        if not report_path:
            print("[ERROR][VinSolutions] Report path not specified in config.")
            return None
            
        try:
            # Navigate to the report section (example path)
            print(f"[INFO][VinSolutions] Navigating to report path: {report_path}")
            # Placeholder for actual navigation logic using self.client (e.g., Selenium)
            # self.client.get(f"https://vinsolutions.example.com/{report_path}") 
            # time.sleep(5) # Allow page to load

            # --- Add logic for report naming ---
            print("[INFO][VinSolutions] Listing files in report directory...")
            # Placeholder: Assume self.client has a method list_files that returns 
            # a list of dictionaries like: [{'name': 'report_abc.xlsx', 'date': datetime_obj, 'url': '...'}]
            # Replace with actual implementation based on NovaActClient capabilities
            files = self.client.list_files(report_path) 
            if not files:
                 print(f"[WARN][VinSolutions] No files found in directory: {report_path}")
                 return None

            # Find the latest file based on date (ensure date parsing is correct)
            # Example assumes 'date' is a datetime object or comparable string
            latest_file_info = max(files, key=lambda f: f.get('date', None), default=None)

            if not latest_file_info:
                 print(f"[WARN][VinSolutions] Could not determine the latest file.")
                 return None
                 
            latest_file_name = latest_file_info.get('name')
            latest_file_url = latest_file_info.get('url')
            
            print(f"[INFO][VinSolutions] Latest file identified: {latest_file_name}")
            
            # Check if the latest file has the expected extension
            if latest_file_name and latest_file_url and latest_file_name.endswith(('.csv', '.xlsx')):
                print(f"[INFO][VinSolutions] Downloading latest file: {latest_file_name} from {latest_file_url}")
                # Placeholder: Assume self.client has download_file method
                downloaded_path = self.client.download_file(latest_file_url, save_dir="./nova_downloads") # Example save dir
                print(f"[INFO][VinSolutions] File downloaded to: {downloaded_path}")
                self._logout()
                return downloaded_path
            else:
                print(f"[WARN][VinSolutions] Latest file '{latest_file_name}' does not have a supported extension (.csv, .xlsx) or URL is missing.")
                self._logout()
                return None
            # --- End logic for report naming ---

        except Exception as e:
            print(f"[ERROR][VinSolutions] Failed during report collection: {e}")
            # Add specific error handling for navigation/download issues
            self._logout()
            return None