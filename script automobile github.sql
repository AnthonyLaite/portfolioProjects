select * from automobiledata order by 3,4; 
select EngineType, CylinderCount, EngineSize, HorsePower, PriceinDollars from automobiledata; 
select EngineSize, avg(PriceinDollars) As AvgPrice from automobiledata
Group BY EngineSize
Order By EngineSize;
SELECT  HorsePower, PriceinDollars, PriceinDollars / HorsePower AS PricePerHorsepower
FROM automobiledata;
SELECT EngineType, CylinderCount, EngineSize, HorsePower, PriceinDollars
FROM automobiledata
WHERE HorsePower > 200 AND PriceinDollars < 25000;
SELECT 
    FuelType, 
    COUNT(*) as VehicleCount
FROM 
    automobiledata
GROUP BY 
    FuelType;
SELECT 
    Aspiration, 
    COUNT(*) as VehicleCount
FROM 
    automobiledata
GROUP BY 
    Aspiration;
    -- Fuel Type Analysis
    SELECT 
    CylinderCount, 
    EngineSize, 
    HorsePower
FROM 
    automobiledata
ORDER BY 
    HorsePower DESC;
SELECT 
    FuelType,
    AVG(CityMileage) AS AvgCityMileage,
    AVG(HighwayMileage) AS AvgHighwayMileage,
    ((AVG(HighwayMileage) - AVG(CityMileage)) / AVG(CityMileage)) * 100 AS PercentageDifference
FROM 
    automobiledata
GROUP BY 
    FuelType;
    SELECT 
    Brandname,
    AVG(CityMileage) AS AvgCityMileage,
    AVG(HighwayMileage) AS AvgHighwayMileage,
    AVG(HighwayMileage) - AVG(CityMileage) AS MileageDifference
FROM 
   automobiledata
GROUP BY 
   Brandname;
   SELECT 
    BrandName,
    AVG(HorsePower) AS AvgHorsePower,
    AVG(PriceinDollars) AS AvgPriceinDollars
FROM 
    automobiledata
GROUP BY 
    BrandName;
    SELECT 
    FuelType,
    COUNT(*) AS NumberOfVehicles,
    AVG(HorsePower) AS AvgHorsePower,
    STDDEV(HorsePower) AS StdDevHorsePower,
    AVG(PriceinDollars) AS AvgPriceinDollars,
    STDDEV(PriceinDollars) AS StdDevPriceinDollars,
    (STDDEV(HorsePower) / AVG(HorsePower)) * 100 AS CoefficientOfVariationHorsePower,
    (STDDEV(PriceinDollars) / AVG(PriceinDollars)) * 100 AS CoefficientOfVariationPrice
FROM 
    automobiledata
GROUP BY 
    FuelType;
    SELECT 
    Design,
    AVG(EngineSize) AS AvgEngineSize,
    AVG(HorsePower) AS AvgHorsePower,
    AVG(HorsePower) / AVG(EngineSize) AS PerformanceIndex
FROM 
    automobiledata
GROUP BY 
    Design;