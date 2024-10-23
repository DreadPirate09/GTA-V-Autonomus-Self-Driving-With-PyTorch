using System;
using System.IO;
using GTA;
using GTA.Math;

public class SpeedLogger : Script
{
    private StreamWriter logFile;
    private int interval = 100; 
    private int initDelay = 5000;
    private int elapsedTime = 0;
    private bool isInitialized = false;
    private Ped playerPed;
    private int logTime = 0;

    public SpeedLogger()
    {
        Tick += OnTick;
        Aborted += OnAborted;
        SetupLogging();
    }

    private void SetupLogging()
    {
        try
        {
            string logPath = "C:\\GitRepo1\\PyTorch-Gta-Self-Drive\\PyTorch-Explore-Models\\VehicleSpeedLog.txt"; // use your path here of course
            logFile = new StreamWriter(logPath, true);
            logFile.AutoFlush = true;
        }
        catch (Exception ex)
        {
            GTA.UI.Notification.Show("Error setting up logging: " + ex.Message); 
        }
    }

    private void LogSpeed()
    {
        try
        {
            if (playerPed != null && playerPed.IsInVehicle())
            {
                // logFile.WriteLine("Player in vehicle");
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
                logFile.WriteLine("0");
            }
        }
        catch (Exception ex)
        {
            logFile.WriteLine("Some error caught: " + ex.Message);
            logFile.WriteLine(ex.StackTrace);
            GTA.UI.Notification.Show("Some error caught: " + ex.Message);
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

    private void OnAborted(object sender, EventArgs e)
    {
        if (logFile != null)
        {
            logFile.Close();
        }
    }
}
