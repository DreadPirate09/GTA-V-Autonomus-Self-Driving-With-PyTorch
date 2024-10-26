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

        // Get all vehicles currently in the world
        Vehicle[] vehicles = World.GetAllVehicles();

        if (e.KeyCode == Keys.T)  // Press "T" to trigger
        {
                   // Iterate through each vehicle
            foreach (Vehicle vehicle in vehicles)
            {
                // Check if the vehicle is not the player's vehicle
                if (!vehicle.IsPlayerVehicle())
                {
                    // Mark the vehicle as no longer needed to free memory
                    vehicle.MarkAsNoLongerNeeded();

                    // Remove the vehicle from the world
                    vehicle.Delete();
                }
            }
            ShowNotification("Removed all the cars!");
        }
    }
}
