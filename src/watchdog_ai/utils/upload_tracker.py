"""
Upload history tracking system for CSV imports.
"""

import logging
import pandas as pd
import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import hashlib
import sqlite3
import uuid

logger = logging.getLogger(__name__)

class UploadRecord:
    """Represents a single upload record."""
    
    def __init__(self, 
                file_name: str,
                file_size: int,
                upload_time: datetime,
                dealer_id: Optional[str] = None,
                user_id: Optional[str] = None,
                status: str = "success",
                error: Optional[str] = None,
                row_count: Optional[int] = None,
                column_count: Optional[int] = None,
                file_hash: Optional[str] = None,
                metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize an upload record.
        
        Args:
            file_name: Name of the uploaded file
            file_size: Size of the file in bytes
            upload_time: Time of upload
            dealer_id: Optional dealer ID
            user_id: Optional user ID
            status: Upload status ('success', 'error', 'warning')
            error: Optional error message
            row_count: Optional number of rows in the file
            column_count: Optional number of columns in the file
            file_hash: Optional hash of the file content
            metadata: Optional additional metadata
        """
        self.id = str(uuid.uuid4())
        self.file_name = file_name
        self.file_size = file_size
        self.upload_time = upload_time
        self.dealer_id = dealer_id
        self.user_id = user_id
        self.status = status
        self.error = error
        self.row_count = row_count
        self.column_count = column_count
        self.file_hash = file_hash
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "file_name": self.file_name,
            "file_size": self.file_size,
            "upload_time": self.upload_time.isoformat(),
            "dealer_id": self.dealer_id,
            "user_id": self.user_id,
            "status": self.status,
            "error": self.error,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "file_hash": self.file_hash,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UploadRecord':
        """Create from dictionary."""
        record = cls(
            file_name=data["file_name"],
            file_size=data["file_size"],
            upload_time=datetime.fromisoformat(data["upload_time"]),
            dealer_id=data.get("dealer_id"),
            user_id=data.get("user_id"),
            status=data.get("status", "success"),
            error=data.get("error"),
            row_count=data.get("row_count"),
            column_count=data.get("column_count"),
            file_hash=data.get("file_hash"),
            metadata=data.get("metadata", {})
        )
        record.id = data.get("id", record.id)
        return record


class UploadTracker:
    """
    Tracks file upload history and provides search and filtering capabilities.
    """
    
    def __init__(self, db_path: str = "data/uploads/upload_history.db"):
        """
        Initialize the upload tracker.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize database
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize the SQLite database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create uploads table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS uploads (
                id TEXT PRIMARY KEY,
                file_name TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                upload_time TEXT NOT NULL,
                dealer_id TEXT,
                user_id TEXT,
                status TEXT NOT NULL,
                error TEXT,
                row_count INTEGER,
                column_count INTEGER,
                file_hash TEXT,
                metadata TEXT
            )
            ''')
            
            # Create index on upload_time for faster queries
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_upload_time ON uploads (upload_time)')
            
            conn.commit()
            conn.close()
            
            logger.info(f"Initialized upload history database at {self.db_path}")
        except Exception as e:
            logger.error(f"Error initializing upload history database: {str(e)}")
    
    def track_upload(self, 
                   file_name: str,
                   file_size: int,
                   dealer_id: Optional[str] = None,
                   user_id: Optional[str] = None,
                   df: Optional[pd.DataFrame] = None,
                   file_content: Optional[bytes] = None,
                   status: str = "success",
                   error: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> UploadRecord:
        """
        Track a file upload.
        
        Args:
            file_name: Name of the uploaded file
            file_size: Size of the file in bytes
            dealer_id: Optional dealer ID
            user_id: Optional user ID
            df: Optional DataFrame for row/column counts
            file_content: Optional file content for hashing
            status: Upload status ('success', 'error', 'warning')
            error: Optional error message
            metadata: Optional additional metadata
            
        Returns:
            UploadRecord object
        """
        # Get row and column counts if DataFrame is provided
        row_count = len(df) if df is not None else None
        column_count = len(df.columns) if df is not None else None
        
        # Generate file hash if content is provided
        file_hash = None
        if file_content is not None:
            file_hash = hashlib.md5(file_content).hexdigest()
        
        # Create record
        record = UploadRecord(
            file_name=file_name,
            file_size=file_size,
            upload_time=datetime.now(),
            dealer_id=dealer_id,
            user_id=user_id,
            status=status,
            error=error,
            row_count=row_count,
            column_count=column_count,
            file_hash=file_hash,
            metadata=metadata
        )
        
        # Save to database
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT INTO uploads (
                id, file_name, file_size, upload_time, dealer_id, user_id,
                status, error, row_count, column_count, file_hash, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                record.id,
                record.file_name,
                record.file_size,
                record.upload_time.isoformat(),
                record.dealer_id,
                record.user_id,
                record.status,
                record.error,
                record.row_count,
                record.column_count,
                record.file_hash,
                json.dumps(record.metadata)
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Tracked upload: {record.file_name} ({record.id})")
            
        except Exception as e:
            logger.error(f"Error tracking upload: {str(e)}")
        
        return record
    
    def get_upload_history(self, 
                         limit: int = 100,
                         offset: int = 0,
                         dealer_id: Optional[str] = None,
                         user_id: Optional[str] = None,
                         status: Optional[str] = None,
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None) -> List[UploadRecord]:
        """
        Get upload history with filtering options.
        
        Args:
            limit: Maximum number of records to return
            offset: Number of records to skip
            dealer_id: Optional filter by dealer ID
            user_id: Optional filter by user ID
            status: Optional filter by status
            start_date: Optional filter by start date
            end_date: Optional filter by end date
            
        Returns:
            List of UploadRecord objects
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Return rows as dictionaries
            cursor = conn.cursor()
            
            # Build query
            query = "SELECT * FROM uploads WHERE 1=1"
            params = []
            
            if dealer_id:
                query += " AND dealer_id = ?"
                params.append(dealer_id)
            
            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)
            
            if status:
                query += " AND status = ?"
                params.append(status)
            
            if start_date:
                query += " AND upload_time >= ?"
                params.append(start_date.isoformat())
            
            if end_date:
                query += " AND upload_time <= ?"
                params.append(end_date.isoformat())
            
            # Add ordering and limits
            query += " ORDER BY upload_time DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            # Execute query
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            # Convert to UploadRecord objects
            records = []
            for row in rows:
                row_dict = dict(row)
                
                # Parse metadata JSON
                if row_dict.get("metadata"):
                    try:
                        row_dict["metadata"] = json.loads(row_dict["metadata"])
                    except:
                        row_dict["metadata"] = {}
                
                records.append(UploadRecord.from_dict(row_dict))
            
            conn.close()
            return records
            
        except Exception as e:
            logger.error(f"Error getting upload history: {str(e)}")
            return []
    
    def get_upload_stats(self, 
                       dealer_id: Optional[str] = None,
                       user_id: Optional[str] = None,
                       days: int = 30) -> Dict[str, Any]:
        """
        Get upload statistics.
        
        Args:
            dealer_id: Optional filter by dealer ID
            user_id: Optional filter by user ID
            days: Number of days to include in stats
            
        Returns:
            Dictionary with upload statistics
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Calculate start date
            start_date = (datetime.now() - pd.Timedelta(days=days)).isoformat()
            
            # Build query
            query = """
            SELECT 
                COUNT(*) as total_uploads,
                SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successful_uploads,
                SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as failed_uploads,
                SUM(CASE WHEN status = 'warning' THEN 1 ELSE 0 END) as warning_uploads,
                SUM(file_size) as total_size,
                AVG(file_size) as avg_size,
                MAX(file_size) as max_size,
                MIN(upload_time) as first_upload,
                MAX(upload_time) as last_upload,
                COUNT(DISTINCT dealer_id) as unique_dealers,
                COUNT(DISTINCT user_id) as unique_users
            FROM uploads
            WHERE upload_time >= ?
            """
            params = [start_date]
            
            if dealer_id:
                query += " AND dealer_id = ?"
                params.append(dealer_id)
            
            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)
            
            # Execute query
            cursor.execute(query, params)
            row = cursor.fetchone()
            
            # Get file type distribution
            file_types_query = """
            SELECT 
                SUBSTR(file_name, INSTR(file_name, '.') + 1) as file_ext,
                COUNT(*) as count
            FROM uploads
            WHERE upload_time >= ?
            """
            
            if dealer_id:
                file_types_query += " AND dealer_id = ?"
            
            if user_id:
                file_types_query += " AND user_id = ?"
            
            file_types_query += " GROUP BY file_ext ORDER BY count DESC"
            
            cursor.execute(file_types_query, params)
            file_types = {ext: count for ext, count in cursor.fetchall()}
            
            # Get daily upload counts
            daily_query = """
            SELECT 
                DATE(upload_time) as date,
                COUNT(*) as count
            FROM uploads
            WHERE upload_time >= ?
            """
            
            if dealer_id:
                daily_query += " AND dealer_id = ?"
            
            if user_id:
                daily_query += " AND user_id = ?"
            
            daily_query += " GROUP BY DATE(upload_time) ORDER BY date"
            
            cursor.execute(daily_query, params)
            daily_counts = {date: count for date, count in cursor.fetchall()}
            
            conn.close()
            
            # Format results
            stats = {
                "total_uploads": row[0] or 0,
                "successful_uploads": row[1] or 0,
                "failed_uploads": row[2] or 0,
                "warning_uploads": row[3] or 0,
                "total_size_bytes": row[4] or 0,
                "avg_size_bytes": row[5] or 0,
                "max_size_bytes": row[6] or 0,
                "first_upload": row[7],
                "last_upload": row[8],
                "unique_dealers": row[9] or 0,
                "unique_users": row[10] or 0,
                "file_types": file_types,
                "daily_counts": daily_counts,
                "success_rate": (row[1] / row[0] * 100) if row[0] and row[1] else 0
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting upload stats: {str(e)}")
            return {
                "error": str(e),
                "total_uploads": 0,
                "successful_uploads": 0,
                "failed_uploads": 0
            }
    
    def get_upload_by_id(self, upload_id: str) -> Optional[UploadRecord]:
        """
        Get an upload record by ID.
        
        Args:
            upload_id: Upload record ID
            
        Returns:
            UploadRecord object or None if not found
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM uploads WHERE id = ?", (upload_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            row_dict = dict(row)
            
            # Parse metadata JSON
            if row_dict.get("metadata"):
                try:
                    row_dict["metadata"] = json.loads(row_dict["metadata"])
                except:
                    row_dict["metadata"] = {}
            
            conn.close()
            return UploadRecord.from_dict(row_dict)
            
        except Exception as e:
            logger.error(f"Error getting upload by ID: {str(e)}")
            return None
    
    def update_upload_status(self, upload_id: str, 
                           status: str,
                           error: Optional[str] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update the status of an upload record.
        
        Args:
            upload_id: Upload record ID
            status: New status
            error: Optional error message
            metadata: Optional metadata to update
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get existing record
            record = self.get_upload_by_id(upload_id)
            if not record:
                logger.warning(f"Upload record not found: {upload_id}")
                return False
            
            # Update record
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Update status and error
            query = "UPDATE uploads SET status = ?, error = ?"
            params = [status, error]
            
            # Update metadata if provided
            if metadata:
                # Merge with existing metadata
                updated_metadata = {**record.metadata, **metadata}
                query += ", metadata = ?"
                params.append(json.dumps(updated_metadata))
            
            query += " WHERE id = ?"
            params.append(upload_id)
            
            cursor.execute(query, params)
            conn.commit()
            conn.close()
            
            logger.info(f"Updated upload status: {upload_id} -> {status}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating upload status: {str(e)}")
            return False
    
    def delete_upload_record(self, upload_id: str) -> bool:
        """
        Delete an upload record.
        
        Args:
            upload_id: Upload record ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM uploads WHERE id = ?", (upload_id,))
            conn.commit()
            conn.close()
            
            logger.info(f"Deleted upload record: {upload_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting upload record: {str(e)}")
            return False
