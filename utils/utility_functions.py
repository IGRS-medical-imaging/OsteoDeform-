import logging
import os
import shutil
import torch


def get_logger(log_dir):
    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    stream_handler = logging.StreamHandler()
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(
        os.path.join(log_dir, os.path.basename("log.txt"))
    )
    logger.addHandler(file_handler)
    return logger

def save_checkpoint(state_dict, is_best, epoch, output_dir, filename, logger):
    """Save checkpoint.
    """
    if epoch > 1:
        prev_ckpt = (
            output_dir + filename + "_%03d" % (epoch - 1) + ".pth.tar"
        )
        if os.path.exists(prev_ckpt):
            os.remove(prev_ckpt)
    torch.save(state_dict, output_dir + filename + "_%03d" %
               epoch + ".pth.tar")
    print(output_dir + filename + "_%03d" % epoch + ".pth.tar")
    if is_best:
        if logger is not None:
            logger.info("Saving new best model")

        shutil.copyfile(
            output_dir + filename + "_%03d" % epoch + ".pth.tar",
            output_dir + filename + "_best.pth.tar",
        )


def snapshot_files(list_of_filenames, log_dir):
    """Snapshot list of files in current run state to the log directory.
    """
    snap_dir = os.path.join(log_dir, "snapshots")
    os.makedirs(snap_dir, exist_ok=True)
    for filename in list_of_filenames:
        out_name = os.path.basename(filename)
        shutil.copy2(filename, os.path.join(snap_dir, out_name))

