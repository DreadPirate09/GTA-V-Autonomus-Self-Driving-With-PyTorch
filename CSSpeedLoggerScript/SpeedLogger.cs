using System;
using System.IO;
using GTA;
using GTA.Math;

public class SpeedLogger : Script
{
    private StreamWriter logFile;
    private int interval = 100; // Logging interval in milliseconds
    private int initDelay = 5000; // Delay in milliseconds before starting logging
    private int elapsedTime = 0;
    private bool isInitialized = false;
    private Ped playerPed;
    private int logTime = 0;

    public SpeedLogger()
    {
        Tick += OnTick;
        SetupLogging();
    }

    private void SetupLogging()
    {
        try
        {
            string logPath = "C:\\Program Files\\Epic Games\\GTAV\\scripts\\VehicleSpeedLog.txt";
            logFile = new StreamWriter(logPath, true); // Append to the log file
            logFile.AutoFlush = true;
        }
        catch (Exception ex)
        {
            logFile.WriteLine("Some error caught: " + ex.Message);
            UI.Notify("Error setting up logging: " + ex.Message);
            GTA.UI.Notify("Error setting up logging: " + ex.Message);
        }
    }

    private void LogSpeed()
    {
        try
        {
            // Check if the player's character is in a vehicle
            if (playerPed != null && playerPed.IsInVehicle())
            {
                logFile.WriteLine("Player in vehicle");
                // Get the player's current vehicle
                var playerVehicle = playerPed.CurrentVehicle;
                if (playerVehicle != null && playerVehicle.Exists())
                {
                    // Calculate speed in km/h
                    float speed = playerVehicle.Speed * 3.6f;
                    // Log the speed
                    string logEntry = speed.ToString("F2");
                    logFile.WriteLine(logEntry);
                }
                else
                {
                    logFile.WriteLine("Player vehicle is null or does not exist");
                }
            }
            else
            {
                logFile.WriteLine("Player not in vehicle");
            }
        }
        catch (Exception ex)
        {
            logFile.WriteLine("Some error caught: " + ex.Message);
            logFile.WriteLine(ex.StackTrace);
            GTA.UI.Notify("Some error caught: " + ex.Message);
        }
    }

    private void OnTick(object sender, EventArgs e)
    {
        elapsedTime += (int)(Game.LastFrameTime * 1000);

        if (!isInitialized && elapsedTime >= initDelay)
        {
            logFile.WriteLine("Starting initialization");
            isInitialized = true;
            logFile.WriteLine("Initialization complete. Starting logging.");
        }

        if (isInitialized)
        {
            logTime += (int)(Game.LastFrameTime * 1000);
            if (logTime >= interval)
            {
                LogSpeed();
                logTime = 0;
            }
        }

        playerPed = Game.Player.Character;
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            if (logFile != null)
            {
                logFile.Close();
            }
        }
        base.Dispose(disposing);
    }
}
