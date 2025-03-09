import wandb
import pytorch_lightning as pl
import time
import socket
import logging
from pytorch_lightning.loggers import WandbLogger

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DynamicWandbLogger(WandbLogger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_offline = False  # Track if in offline mode

    def _check_internet(self):
        """Check if the internet is available by connecting to Google DNS."""
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True
        except OSError:
            return False

    def log_metrics(self, metrics, step=None):
        """Dynamically switch between online and offline modes based on network status."""
        if self._is_offline:
            try:
                if self._check_internet():
                    logger.info("[INFO] Internet restored! Resuming online logging.")
                    self.experiment = wandb.init(resume=True)
                    self._is_offline = False
            except Exception as e:
                logger.warning(f"[WARNING] Failed to resume online mode: {e}")

        if not self._is_offline:
            try:
                super().log_metrics(metrics, step)
            except wandb.errors.CommError:
                logger.warning("[WARNING] Internet lost! Switching to offline mode.")
                wandb.finish()
                self.experiment = wandb.init(mode="offline")
                self._is_offline = True

    def auto_sync_logs(self):
        """Try to sync offline logs after training, retrying every 2 seconds if internet is down."""
        while not self._check_internet():
            logger.warning("[WARNING] No internet. Retrying in 2 seconds...")
            time.sleep(2)
        
        logger.info("[INFO] Internet detected. Syncing offline logs...")
        try:
            wandb.finish()  # Close any open session
            wandb.sync("wandb/offline-run-*")
            logger.info("[INFO] Offline logs successfully synced!")
        except Exception as e:
            logger.error(f"[ERROR] Failed to sync logs: {e}")


