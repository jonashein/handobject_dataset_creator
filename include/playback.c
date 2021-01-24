// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stdio.h>
#include <malloc.h>
#include <k4a/k4a.h>
#include <k4arecord/playback.h>

typedef struct
{
    const char *filename;
    k4a_playback_t handle;
    k4a_record_configuration_t record_config;
    k4a_capture_t capture;
} recording_t;

static uint64_t earliest_capture_timestamp(k4a_capture_t capture)
{
    uint64_t min_timestamp = (uint64_t)-1;
    k4a_image_t images[3];
    images[0] = k4a_capture_get_color_image(capture);
    images[1] = k4a_capture_get_depth_image(capture);
    images[2] = k4a_capture_get_ir_image(capture);

    for (int i = 0; i < 3; i++)
    {
        if (images[i] != NULL)
        {
            uint64_t timestamp = k4a_image_get_device_timestamp_usec(images[i]);
            if (timestamp < min_timestamp)
            {
                min_timestamp = timestamp;
            }
            k4a_image_release(images[i]);
            images[i] = NULL;
        }
    }

    return min_timestamp;
}

static void print_capture_info(recording_t *file)
{
    k4a_image_t images[3];
    images[0] = k4a_capture_get_color_image(file->capture);
    images[1] = k4a_capture_get_depth_image(file->capture);
    images[2] = k4a_capture_get_ir_image(file->capture);

    printf("%-32s", file->filename);
    for (int i = 0; i < 3; i++)
    {
        if (images[i] != NULL)
        {
            uint64_t timestamp = k4a_image_get_device_timestamp_usec(images[i]);
            printf("  %7ju usec", timestamp);
            k4a_image_release(images[i]);
            images[i] = NULL;
        }
        else
        {
            printf("  %12s", "");
        }
    }
    printf("\n");
}