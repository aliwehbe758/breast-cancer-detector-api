from app import db
import base64

class PredictionH(db.Model):
    __tablename__ = 'prediction_h'
    id = db.Column(db.Integer, primary_key=True)
    uploaded_folder_name = db.Column(db.String(255), nullable=False)
    total_images = db.Column(db.BigInteger, nullable=False)
    benign_count = db.Column(db.BigInteger, nullable=False)
    malignant_count = db.Column(db.BigInteger, nullable=False)
    normal_count = db.Column(db.BigInteger, nullable=False)
    date = db.Column(db.DateTime, nullable=False)
    pretrained_model_id = db.Column(db.BigInteger, db.ForeignKey('pretrained_models.id'), nullable=False)
    details = db.relationship('PredictionD', backref='prediction_h', lazy=True, cascade="all, delete-orphan")

    def to_dict1(self):
        return {
            'id': self.id,
            'uploaded_folder_name': self.uploaded_folder_name,
            'total_images': self.total_images,
            'benign_count': self.benign_count,
            'malignant_count': self.malignant_count,
            'normal_count': self.normal_count,
            'date': self.date,
            'pretrained_model_id': self.pretrained_model_id,
            'pretrained_model_name': self.pretrained_models.name,
            'details': [detail.to_dict() for detail in self.details]
        }
    def to_dict2(self):
        return {
            'id': self.id,
            'uploaded_folder_name': self.uploaded_folder_name,
            'total_images': self.total_images,
            'benign_count': self.benign_count,
            'malignant_count': self.malignant_count,
            'normal_count': self.normal_count,
            'pretrained_model_id': self.pretrained_model_id,
            'pretrained_model_name': self.pretrained_models.name,
            'date': self.date
        }

class PredictionD(db.Model):
    __tablename__ = 'prediction_d'
    id = db.Column(db.BigInteger, primary_key=True)
    file_name = db.Column(db.String(255), nullable=False)
    prediction = db.Column(db.String(255), nullable=False)
    probability = db.Column(db.Double, nullable=False)
    image = db.Column(db.LargeBinary, nullable=False)
    grad_cam = db.Column(db.LargeBinary, nullable=False)
    prediction_h_id = db.Column(db.BigInteger, db.ForeignKey('prediction_h.id'), nullable=False)

    def to_dict(self):
        return {
            'id': self.id,
            'file_name': self.file_name,
            'prediction': self.prediction,
            'probability': self.probability,
            'image': base64.b64encode(self.image).decode('utf-8'),
            'grad_cam': base64.b64encode(self.grad_cam).decode('utf-8')
        }

class PretrainedModel(db.Model):
    # inference_models
    __tablename__ = 'pretrained_models'
    id = db.Column(db.BigInteger, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text, nullable=True)
    pth_file_name = db.Column(db.String(255), nullable=False)
    model_py_file_name = db.Column(db.String(255), nullable=False)
    params_py_file_name = db.Column(db.String(255), nullable=False)
    folder_name = db.Column(db.String(255), nullable=False)
    upload_date = db.Column(db.DateTime, nullable=False)
    predictions = db.relationship('PredictionH', backref='pretrained_models', lazy=True, cascade="all, delete-orphan")

    def __init__(self, id, name, description, pth_file_name, model_py_file_name, params_py_file_name, folder_name, upload_date):
        self.id = id
        self.name = name
        self.description = description
        self.pth_file_name = pth_file_name
        self.model_py_file_name = model_py_file_name
        self.params_py_file_name = params_py_file_name
        self.folder_name = folder_name
        self.upload_date = upload_date

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'pth_file_name': self.pth_file_name,
            'model_py_file_name': self.model_py_file_name,
            'params_py_file_name': self.params_py_file_name,
            'folder_name': self.folder_name,
            'upload_date': self.upload_date.isoformat()
        }