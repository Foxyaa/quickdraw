using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using Unity.Sentis;
using Random = UnityEngine.Random;
//using System.Linq;

public class Paint : MonoBehaviour
{
    [SerializeField] private int _textureSize = 128;
    [SerializeField] private TextureWrapMode _textureWrapMode;
    [SerializeField] private FilterMode _filterMode;
    [SerializeField] private Texture2D _texture;
    [SerializeField] private Material _material;
    [SerializeField] private Camera _camera;
    [SerializeField] private Collider _collider;
    //[SerializeField] private Color _color;  //#5B6EAB #0A8584
    [SerializeField] private Color[] colorsBrush = new Color[]
    {
        new Color32(247, 160, 114, 255), // #f7a072 
        new Color32(181, 226, 250, 255), // #b5e2fa
        new Color32(56, 116, 244, 255)  // #3874f4
    };
    [SerializeField] private Color randomColorBrush;
    [SerializeField] private Color[] colorsBackg = new Color[]
    {
        new Color32(237, 237, 233, 255), // #edede9
        new Color32(237, 229, 222, 255), // #edede9
        new Color32(244, 243, 238, 255),  // #f4f3ee
        new Color32(237, 225, 222, 255),  // #ede1de
        new Color32(243, 233, 234, 255)  // #f3e9ea
    };
    [SerializeField] private Color randomColorBackg;
    [SerializeField] private int _brushSize = 5; //радиус

    [SerializeField] private float _timeRemaining = 15f;

    private Model model;
    private const string modelPath = "Assets/Scripts/detect_image_model.onnx";
    private IWorker engine;
    //private Worker engine;

    void OnValidate() {
        if (_texture == null) {
            _texture = new Texture2D(_textureSize, _textureSize);
        }
        if (_texture.width != _textureSize) {
            _texture.Reinitialize(_textureSize, _textureSize);
        }
        _texture.wrapMode = _textureWrapMode;
        _texture.filterMode = _filterMode;
        _material.mainTexture = _texture;
        _texture.Apply();
    }
    // Start is called before the first frame update
    void Start()
    {
        randomColorBrush = colorsBrush[Random.Range(0, colorsBrush.Length)];
        randomColorBackg = colorsBackg[Random.Range(0, colorsBackg.Length)];
        for (int y = 0; y < _textureSize; y++) {
            for (int x = 0; x < _textureSize; x++) {
                 _texture.SetPixel(x, y, randomColorBackg);
            }
        }
        _texture.Apply();

        StartCoroutine(CountdownAndSave(_timeRemaining));
    }

    // Update is called once per frame
    private void Update()
    {
        if (Input.GetMouseButton(0)) {
            Ray ray = _camera.ScreenPointToRay(Input.mousePosition);

            RaycastHit hit;
            if (_collider.Raycast(ray, out hit, 100f)) {
                int rayX = (int)(hit.textureCoord.x * _textureSize);
                int rayY = (int)(hit.textureCoord.y * _textureSize);
                
                DrawCircleBrush(rayX, rayY);
                _texture.Apply();
            }
        }
    }
    void DrawCircleBrush(int rayX, int rayY) {
        for (int y = 0; y < _brushSize; y++) {
            for (int x = 0; x < _brushSize; x++) {
                float x2 = Mathf.Pow(x - _brushSize/2, 2);
                float y2 = Mathf.Pow(y - _brushSize/2, 2);
                float r2 = Mathf.Pow(_brushSize/2 - 0.5f, 2);

                if (x2 + y2 < r2) {
                    _texture.SetPixel(rayX + x - _brushSize, rayY + y - _brushSize, randomColorBrush);
                }
            }
        }
    }
    private IEnumerator CountdownAndSave(float duration)
    {
        yield return new WaitForSeconds(duration);
        SaveToPNG(); // Вызов метода сохранения в PNG
    }
    void SaveToPNG()
    {
        Texture2D texture = _material.mainTexture as Texture2D;
        
        if (texture == null)
        {
            Debug.LogError("Текстура не найдена или не является Texture2D.");
            return;
        }
        // Масштабируем текстуру до 128x128
        Texture2D resizedTexture = ResizeTexture(texture, 128, 128);
        byte[] pngData = resizedTexture.EncodeToPNG();
        if (pngData != null)
    {
        string path = "Assets/Resources/pictures_in_homes/image.png";
        System.IO.File.WriteAllBytes(path, pngData);
        Debug.Log("Изображение успешно сохранено по пути: " + path);
    }
    else
    {
        Debug.LogError("Не удалось закодировать текстуру в PNG.");
    }
        SendToModel();
    }
    
    Texture2D ResizeTexture(Texture2D originalTexture, int width, int height){
        RenderTexture renderTex = RenderTexture.GetTemporary(width, height);
        RenderTexture.active = renderTex;

        Graphics.Blit(originalTexture, renderTex);

        Texture2D resizedTexture = new Texture2D(width, height, TextureFormat.RGBA32, false);
        resizedTexture.ReadPixels(new Rect(0, 0, width, height), 0, 0);
        resizedTexture.Apply();

        RenderTexture.active = null;
        RenderTexture.ReleaseTemporary(renderTex);

        return resizedTexture;
    }
    void SendToModel() {
        //model = ModelLoader.Load(modelPath);
        //engine = new Worker(model, BackendType.GPUCompute);
        model = ModelLoader.Load( Application.streamingAssetsPath + "/detect_image_model.sentis" );//ModelLoader.Load(modelPath);
        engine = WorkerFactory.CreateWorker(BackendType.GPUCompute, model);
        Texture2D inputImage = new Texture2D(128, 128);
        byte[] fileData = System.IO.File.ReadAllBytes("Assets/Resources/pictures_in_homes/image.png");
        inputImage.LoadImage(fileData);
        // Преобразование текстуры в градации серого
        //Texture2D grayImage = ConvertToGrayscale(inputImage);

        // Подготовка данных для модели
        
        Texture2D inputData = ConvertToGrayscale(inputImage);
        TensorFloat inputTensor = PrepareInputData(inputData);
        //TensorFloat inputTensor = new TensorFloat(new TensorShape(1, 128, 128, 1), inputData);
        // Выполнение предсказания
        //engine.Schedule(inputTensor);
        //Tensor<float> outputTensor = engine.PeekOutput() as Tensor<float>;
        TensorFloat outputTensor = engine.Execute(inputTensor).PeekOutput() as TensorFloat;
        // Обработка результата
        ProcessResult(outputTensor);

        // Освобождение ресурсов
        inputTensor.Dispose();
        outputTensor.Dispose();
    }
    Texture2D ConvertToGrayscale(Texture2D original)
    {
        Texture2D grayTexture = new Texture2D(original.width, original.height);
        for (int y = 0; y < original.height; y++)
        {
            for (int x = 0; x < original.width; x++)
            {
                Color color = original.GetPixel(x, y);
                float grayValue = color.grayscale; // Среднее значение для градаций серого
                grayTexture.SetPixel(x, y, new Color(grayValue, grayValue, grayValue));
            }
        }
        grayTexture.Apply();
        return grayTexture;
    }
    
    TensorFloat PrepareInputData(Texture2D grayImage)
    {
        float[] inputData = new float[128 * 128 * 1]; // 128x128 и 1 канал
        for (int y = 0; y < grayImage.height; y++)
        {
            for (int x = 0; x < grayImage.width; x++)
            {
                Color color = grayImage.GetPixel(x, y);
                inputData[y * grayImage.width + x] = color.r; // Используем только один канал
            }
        }

        return new TensorFloat(new TensorShape(1, 128, 128, 1), inputData);//new Tensor<float>(new TensorShape(1, 128, 128, 1), inputData); // Формат: (batch, height, width, channels)
    }
    void ProcessResult(TensorFloat outputTensor)
    {
        //outputTensor.ReadbackRequest();
        //var result = outputTensor.ReadbackAndClone();
        //Debug.Log("Model output: " + result);
        try{
        outputTensor.MakeReadable();
        //outputTensor.PrintDataPart(10);
        string[] classes = {
        "birthday cake", "clock", "cactus", "flamingo", "tornado", "train",
        "octopus", "lighthouse", "mountain", "palm tree", "sailboat", 
        "wine glass", "pencil", "butterfly"
        };

    // Проверка размерности выходного тензора
    if (outputTensor.shape[1] != classes.Length)
    {
        Debug.LogError("Размерность выходного тензора не соответствует количеству классов.");
        return;
    }
    float[] tensorData = outputTensor.ToReadOnlyArray();
    // Найти индекс наибольшего значения
    int maxIndex = 0;
    float maxValue = float.MinValue;
    Debug.Log($"Инициализация мин. значения: {maxValue}");
    for (int i = 0; i < tensorData.Length; i++)
    {
        if (tensorData[i] > maxValue)
        {
            maxValue = tensorData[i];
            maxIndex = i;
        }
        //Debug.Log($"Все тензоры: {tensorData[i]}, индекс: {i}");
    }

    // Название класса
    string resultClass = classes[maxIndex];

    // Вывод результата в консоль
    //Debug.Log($"Наибольшее значение: {maxValue}");
    //Debug.Log($"Порядковый индекс: {maxIndex}");
    Debug.Log($"Название класса: {resultClass}");
    }
    finally
    {
        // Освобождение ресурсов
        outputTensor.Dispose();
    }
}
    void OnDestroy()
    {
        engine?.Dispose();
    }
}
