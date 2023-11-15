import React, { useState } from 'react';

function ImageUploader(props) {
  const [imagePreview, setImagePreview] = useState(null);

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result);
        props.onImageUpload(reader.result);
      }
      reader.readAsDataURL(file);
    }
  }

  return (
      <div className="image-uploader">
        <input type="file" onChange={handleImageChange} />
        {imagePreview && <img src={imagePreview} alt="Preview" width="100%" />}
      </div>
  );
}

export default ImageUploader;
