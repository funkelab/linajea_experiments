import sys
import os


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python cleanup_checkpoints.py <setup> <skip_by>\n"
              "Filters checkpoints to those with number n*<skip_by>")
        sys.exit(1)

    checkpoint_dir = sys.argv[1]
    skip_by = int(sys.argv[2])

    i = 0
    for checkpoint_name in os.listdir(checkpoint_dir):
        if "_checkpoint_" not in checkpoint_name:
            continue
        num = int(checkpoint_name.split("_checkpoint_")[-1].split(".")[0])
        if num % skip_by == 0:
            print("Keeping checkpoint %d" % num)
        else:
            print("Trashing checkpoint %s" % os.path.join(checkpoint_dir, checkpoint_name))
            os.remove(os.path.join(checkpoint_dir, checkpoint_name))
