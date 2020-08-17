"""
this is a demo for processing video data.
"""

import time
import argparse
import os
import csv

import torch
import cv2

import posenet

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=int, default=101)
parser.add_argument("--scale_factor", type=float, default=1.0)
parser.add_argument("--notxt", action="store_true")
parser.add_argument("--input", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--keypoint", type=str, required=True)
args = parser.parse_args()


def main():
    model = posenet.load_model(args.model)
    model = model.cuda()
    output_stride = model.output_stride

    if args.output:
        if os.path.exists(args.output):
            raise RuntimeError("Output file already exists")
        if not args.output.endswith(".avi"):
            raise RuntimeError("Output file doesn't end with .avi")

    if args.keypoint:
        if os.path.exists(args.keypoint):
            raise RuntimeError("Keypoint file already exists")
        if not args.keypoint.endswith(".csv"):
            raise RuntimeError("Keypoint file doesn't end with .csv")

    filename = args.input

    cap = cv2.VideoCapture(filename)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
    size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(args.output, fourcc, 20.0, size)

    start = time.time()
    frame_count = 0
    data_list = []
    while cap.isOpened():
        res = posenet.read_img_from_cap(
            cap, scale_factor=args.scale_factor, output_stride=output_stride
        )
        if not res:
            break
        input_image, draw_image, output_scale = res

        with torch.no_grad():
            input_image = torch.Tensor(input_image).cuda()

            (
                heatmaps_result,
                offsets_result,
                displacement_fwd_result,
                displacement_bwd_result,
            ) = model(input_image)

            (
                pose_scores,
                keypoint_scores,
                keypoint_coords,
            ) = posenet.decode_multiple_poses(
                heatmaps_result.squeeze(0),
                offsets_result.squeeze(0),
                displacement_fwd_result.squeeze(0),
                displacement_bwd_result.squeeze(0),
                output_stride=output_stride,
                max_pose_detections=10,
                min_pose_score=0.25,
            )

        keypoint_coords *= output_scale

        if args.output:
            draw_image = posenet.draw_skel_and_kp(
                draw_image,
                pose_scores,
                keypoint_scores,
                keypoint_coords,
                min_pose_score=0.25,
                min_part_score=0.25,
            )

            out.write(draw_image)

        if not args.notxt:
            print()
            print("Results for frame: %s" % frame_count)
            for (pose_index, score) in enumerate(pose_scores):
                if score == 0.0:
                    break
                print("Pose #%d, score = %f" % (pose_index, score))
                for keypoint_index, (keypoint_score, keypoint_coords) in enumerate(
                    zip(
                        keypoint_scores[pose_index, :],
                        keypoint_coords[pose_index, :, :],
                    )
                ):
                    print(
                        "Keypoint %s, score = %f, coord = %s"
                        % (
                            posenet.PART_NAMES[keypoint_index],
                            keypoint_score,
                            keypoint_coords,
                        )
                    )

        for (pose_index, score) in enumerate(pose_scores):
            if pose_scores[pose_index] == 0.0:
                break
            csv_dict = {}
            csv_dict["frame"] = frame_count
            csv_dict["score"] = pose_scores[pose_index]
            for keypoint_index, (keypoint_score, keypoint_coords) in enumerate(
                zip(keypoint_scores[pose_index, :], keypoint_coords[pose_index, :, :],)
            ):
                part_name = posenet.PART_NAMES[keypoint_index]
                csv_dict[f"{part_name}_score"] = keypoint_score
                csv_dict[f"{part_name}_x"] = keypoint_coords[0]
                csv_dict[f"{part_name}_y"] = keypoint_coords[1]
            data_list.append(csv_dict)

        frame_count += 1

    if data_list:
        file_path = args.keypoint
        with open(file_path, "w", encoding="utf8", newline="") as output_file:
            csv_writer = csv.DictWriter(output_file, fieldnames=data_list[0].keys())
            csv_writer.writeheader()
            csv_writer.writerows(data_list)

    cap.release()
    print("Average FPS:", frame_count / (time.time() - start))


if __name__ == "__main__":
    main()
