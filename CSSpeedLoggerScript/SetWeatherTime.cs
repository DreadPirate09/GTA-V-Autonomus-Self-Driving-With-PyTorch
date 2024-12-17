using System;
using GTA;
using GTA.Native;
using System.Windows.Forms;

public class WeatherTimeControl : Script
{
    private int timerInterval = 30000; // Interval set to 30 seconds (in milliseconds)
    private int lastExecutionTime = 0; // Tracks when the last weather change happened

    public WeatherTimeControl()
    {
        Tick += OnTick; // Subscribe to the Tick event
        KeyDown += OnKeyDown; // Optional: Allow manual key trigger with "T"
    }

    private void OnTick(object sender, EventArgs e)
    {
        int currentTime = Game.GameTime;

        // Check if 30 seconds have passed since the last execution
        if (currentTime - lastExecutionTime >= timerInterval)
        {
            SetClearWeather();
            lastExecutionTime = currentTime; // Update the last execution time
        }
    }

    private void SetClearWeather()
    {
        // Set weather to clear
        Function.Call((Hash)0x29B487C359E19889, "CLEAR"); // SET_WEATHER_TYPE_NOW

        // Optional: Set time to noon (12:00)
        Function.Call((Hash)0x47C3B5848C3E45D8, 12, 0, 0); // SET_CLOCK_TIME

        // Display notification
        ShowNotification("Weather automatically set to Clear, Time set to Noon!");
    }

    private void OnKeyDown(object sender, KeyEventArgs e)
    {
        if (e.KeyCode == Keys.T) // Manual trigger on pressing "T"
        {
            SetClearWeather();
        }
    }

    private void ShowNotification(string message)
    {
        Function.Call((Hash)0xABA17D7CE615ADBF, "STRING");  // _SET_NOTIFICATION_TEXT_ENTRY
        Function.Call((Hash)0x6C188BE134E074AA, message);   // ADD_TEXT_COMPONENT_SUBSTRING_PLAYER_NAME
        Function.Call((Hash)0x1E6611149DB3DB6B, false, true); // _DRAW_NOTIFICATION
    }
}
