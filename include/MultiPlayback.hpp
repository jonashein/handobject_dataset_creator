// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <chrono>
#include <vector>
#include <iomanip> // std::setw
#include <k4a/k4a.h>
#include <k4arecord/playback.h>

extern "C" {
#include "playback.c"
}

// Maximum delta in timestamps between synchronously taken captures in us
constexpr uint32_t MAX_SYNCHRONIZED_TIMESTAMP_OFFSET = 5000; // in us

class MultiPlayback
{
public:
    // Set up all the devices. Note that the index order isn't necessarily preserved, because we might swap with master
    explicit MultiPlayback(const std::vector<std::string> &playback_files)
    {
        k4a_result_t result;
        file_count = (size_t)playback_files.size();
        playbacks = (recording_t *)malloc(sizeof(recording_t) * file_count);

        if (playbacks == nullptr)
        {
            std::cerr << "Failed to allocate memory for playback (" << sizeof(recording_t) * file_count << " bytes)" << std::endl;
            exit(1);
        }
        memset(playbacks, 0, sizeof(recording_t) * file_count);

        for (size_t i = 0; i < file_count; i++)
        {
            playbacks[i].filename = playback_files.at(i).c_str();

            result = k4a_playback_open(playbacks[i].filename, &playbacks[i].handle);
            if (result != K4A_RESULT_SUCCEEDED)
            {
                std::cerr << "Failed to open file: " << playbacks[i].filename << std::endl;
                exit(2);
            }

            result = k4a_playback_set_color_conversion(playbacks[i].handle, K4A_IMAGE_FORMAT_COLOR_BGRA32);
            if (result != K4A_RESULT_SUCCEEDED)
            {
                std::cerr << "Failed to set color conversion for file: " << playbacks[i].filename << std::endl;
                exit(3);
            }

            result = k4a_playback_get_record_configuration(playbacks[i].handle, &playbacks[i].record_config);
            if (result != K4A_RESULT_SUCCEEDED)
            {
                std::cerr << "Failed to get record configuration for file: " << playbacks[i].filename << std::endl;
                exit(4);
            }

            if (i == 0 && playbacks[i].record_config.wired_sync_mode == K4A_WIRED_SYNC_MODE_MASTER)
            {
                std::cout << "Opened master recording file: " << playbacks[i].filename << std::endl;
            }
            else if (i == 1 && playbacks[i].record_config.wired_sync_mode == K4A_WIRED_SYNC_MODE_SUBORDINATE)
            {
                std::cout << "Opened subordinate recording file: " << playbacks[i].filename << std::endl;
            }
            else
            {
                std::cerr << "Recording file was not in correct main/sub mode: " << playbacks[i].filename << std::endl;
                exit(5);
            }
        }
    }

    k4a_result_t read_calibration(size_t idx, k4a_calibration_t* calibration) {
        if (idx >= file_count)
            return K4A_RESULT_FAILED;
        return k4a_playback_get_calibration(playbacks[idx].handle, calibration);
    }

    uint64_t get_earliest_selected_capture()
    {
        recording_t* earliest = get_earliest_current_capture();
        return earliest_capture_timestamp(earliest->capture);
    }

    uint64_t get_master_timestamp()
    {
        return earliest_capture_timestamp(playbacks[0].capture);
    }

    k4a_result_t open()
    {
        k4a_result_t result = K4A_RESULT_SUCCEEDED;
        for (size_t i = 0; i < file_count; i++)
        {
            k4a_stream_result_t stream_result = k4a_playback_get_next_capture(playbacks[i].handle, &playbacks[i].capture);
            if (stream_result == K4A_STREAM_RESULT_EOF)
            {
                printf("ERROR: Recording file is empty: %s\n", playbacks[i].filename);
                result = K4A_RESULT_FAILED;
                break;
            }
            else if (stream_result == K4A_STREAM_RESULT_FAILED)
            {
                printf("ERROR: Failed to read first capture from file: %s\n", playbacks[i].filename);
                result = K4A_RESULT_FAILED;
                break;
            }
        }
        return result;
    }

    bool seek_timestamp(uint64_t after_timestamp, std::vector<k4a_capture_t> *pCaptures) {
        bool result = true;
        while (result && get_master_timestamp() < after_timestamp) {
            result = next_synchronized_captures(pCaptures);
        }
        return result;
    }

    bool next_synchronized_captures(std::vector<k4a_capture_t> *pCaptures)
    {
        if (pCaptures == nullptr)
            return false;
        pCaptures->clear();

        k4a_result_t result;

        // Select next capture of all recordings
        for (size_t i = 0; i < file_count; i++)
        {
            result = next_capture(playbacks[i]);
            if (result != K4A_RESULT_SUCCEEDED)
                return false;
        }

        // Ensure that selected captures are in sync
        uint64_t time_diff = abs((int64_t)earliest_capture_timestamp(playbacks[0].capture) -
                                 (int64_t)earliest_capture_timestamp(playbacks[1].capture));
        while (time_diff > MAX_SYNCHRONIZED_TIMESTAMP_OFFSET && result == K4A_RESULT_SUCCEEDED)
        {
            // Get the lowest timestamp out of each of the selected captures.
            recording_t* min_playback = get_earliest_current_capture();

            // Update time_diff between selected captures
            result = next_capture(*min_playback);
            time_diff = abs((int64_t)earliest_capture_timestamp(playbacks[0].capture) -
                            (int64_t)earliest_capture_timestamp(playbacks[1].capture));
        }

        if (result == K4A_RESULT_SUCCEEDED)
        {
            for (size_t i = 0; i < file_count; i++)
            {
                if (playbacks[i].capture == nullptr)
                    return false;
                pCaptures->push_back(playbacks[i].capture);
            }
            return true;
        }

        return false;
    }

    void close()
    {
        for (size_t i = 0; i < file_count; i++)
        {
            if (playbacks[i].handle != nullptr)
            {
                k4a_playback_close(playbacks[i].handle);
                playbacks[i].handle = nullptr;
            }
        }
        free(playbacks);
    }

private:
    recording_t *playbacks;
    size_t file_count;

    // Find the lowest timestamp out of each of the current captures.
    recording_t* get_earliest_current_capture() {
        recording_t* result = &playbacks[0];
        uint64_t min_timestamp = earliest_capture_timestamp(playbacks[0].capture);
        for (size_t i = 1; i < file_count; i++)
        {
            if (playbacks[i].capture != nullptr)
            {
                uint64_t timestamp = earliest_capture_timestamp(playbacks[i].capture);
                if (timestamp < min_timestamp)
                {
                    min_timestamp = timestamp;
                    result = &playbacks[i];
                }
            }
        }
        return result;
    }

    static k4a_result_t next_capture(recording_t &recording)
    {
        k4a_result_t result = K4A_RESULT_SUCCEEDED;
        k4a_stream_result_t stream_result;
        bool capture_okay;

        // Read next capture until we get a capture with both color and depth image
        do
        {
            // Release previous capture
            if (recording.capture != nullptr)
            {
                k4a_capture_release(recording.capture);
                recording.capture = nullptr;
            }

            stream_result = k4a_playback_get_next_capture(recording.handle, &recording.capture);
            capture_okay = (stream_result == K4A_STREAM_RESULT_SUCCEEDED);
            if (capture_okay && recording.capture != nullptr)
            {
                k4a_image_t img = k4a_capture_get_color_image(recording.capture);
                if (img == nullptr)
                {
                    capture_okay = false;
                }
                else
                {
                    k4a_image_release(img);
                }

                img = k4a_capture_get_depth_image(recording.capture);
                if (img == nullptr)
                {
                    capture_okay = false;
                }
                else
                {
                    k4a_image_release(img);
                }
            }
        } while (stream_result != K4A_STREAM_RESULT_EOF && !capture_okay);

        if (stream_result == K4A_STREAM_RESULT_EOF)
        {
            printf("Reached end of recording: %s\n", recording.filename);
            result = K4A_RESULT_FAILED;
        }
        else if (stream_result != K4A_STREAM_RESULT_SUCCEEDED)
        {
            printf("Failed to read next capture from file: %s\n", recording.filename);
            result = K4A_RESULT_FAILED;
        }

        return result;
    }
};
