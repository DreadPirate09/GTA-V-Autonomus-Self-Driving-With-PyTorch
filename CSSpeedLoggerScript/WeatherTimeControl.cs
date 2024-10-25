using System;
using GTA;
using GTA.Native;
using System.Windows.Forms; // Import for KeyEventArgs and Keys

public class WeatherTimeControl : Script
{
    public WeatherTimeControl()
    {
        KeyDown += OnKeyDown;  // Use the KeyDown event provided by Script Hook V .NET
    }

    private void OnKeyDown(object sender, KeyEventArgs e)  // Use KeyEventArgs from System.Windows.Forms
    {
        if (e.KeyCode == Keys.T)  // Press "T" to trigger
        {
            // Set weather to clear
            Function.Call((Hash)0x29B487C359E19889, "CLEAR");  // SET_WEATHER_TYPE_NOW

            // Set time to noon (12:00)
            Function.Call((Hash)0x47C3B5848C3E45D8, 12, 0, 0);  // SET_CLOCK_TIME

            // Display notification
            ShowNotification("Weather set to Clear, Time set to Noon!");
        }
    }

    private void ShowNotification(string message)
    {
        // Sets up and displays an on-screen notification using hash values
        Function.Call((Hash)0xABA17D7CE615ADBF, "STRING");  // _SET_NOTIFICATION_TEXT_ENTRY
        Function.Call((Hash)0x6C188BE134E074AA, message);   // ADD_TEXT_COMPONENT_SUBSTRING_PLAYER_NAME
        Function.Call((Hash)0x1E6611149DB3DB6B, false, true); // _DRAW_NOTIFICATION
    }
}
