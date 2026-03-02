"""
房源API工具模块 - 封装所有房源和地标相关的接口调用
"""
import requests
from typing import Optional, Dict, Any, List


class HouseAPITools:
    """房源API工具类，封装所有与房源服务的交互"""
    
    def __init__(self, base_url: str, user_id: str):
        """
        初始化API工具
        
        Args:
            base_url: API基础URL，如 http://IP:8080
            user_id: 用户工号，用于X-User-ID请求头
        """
        self.base_url = base_url.rstrip('/')
        self.user_id = user_id
    
    def _get_headers(self, need_user_id: bool = True) -> Dict[str, str]:
        """获取请求头"""
        headers = {"Content-Type": "application/json"}
        if need_user_id:
            headers["X-User-ID"] = self.user_id
        return headers
    
    def _make_request(self, method: str, endpoint: str, params: Optional[Dict] = None, 
                      need_user_id: bool = True) -> Dict[str, Any]:
        """发起HTTP请求"""
        url = f"{self.base_url}{endpoint}"
        headers = self._get_headers(need_user_id)
        
        try:
            if method.upper() == "GET":
                response = requests.get(url, params=params, headers=headers, timeout=30)
            elif method.upper() == "POST":
                response = requests.post(url, params=params, headers=headers, timeout=30)
            else:
                return {"error": f"不支持的HTTP方法: {method}"}
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"请求失败: {str(e)}"}
    
    # ==================== 房源数据管理 ====================
    
    def init_houses(self) -> Dict[str, Any]:
        """重置房源数据到初始状态"""
        return self._make_request("POST", "/api/houses/init")
    
    # ==================== 地标相关接口 ====================
    
    def get_landmarks(self, category: Optional[str] = None, 
                      district: Optional[str] = None) -> Dict[str, Any]:
        """
        获取地标列表
        
        Args:
            category: 地标类别 subway/company/landmark
            district: 行政区，如 海淀、朝阳
        """
        params = {}
        if category:
            params["category"] = category
        if district:
            params["district"] = district
        return self._make_request("GET", "/api/landmarks", params, need_user_id=False)
    
    def get_landmark_by_name(self, name: str) -> Dict[str, Any]:
        """按名称精确查询地标"""
        return self._make_request("GET", f"/api/landmarks/name/{name}", need_user_id=False)
    
    def search_landmarks(self, q: str, category: Optional[str] = None,
                         district: Optional[str] = None) -> Dict[str, Any]:
        """
        关键词模糊搜索地标
        
        Args:
            q: 搜索关键词
            category: 地标类别
            district: 行政区
        """
        params = {"q": q}
        if category:
            params["category"] = category
        if district:
            params["district"] = district
        return self._make_request("GET", "/api/landmarks/search", params, need_user_id=False)
    
    def get_landmark_by_id(self, landmark_id: str) -> Dict[str, Any]:
        """按地标ID查询详情"""
        return self._make_request("GET", f"/api/landmarks/{landmark_id}", need_user_id=False)
    
    def get_landmark_stats(self) -> Dict[str, Any]:
        """获取地标统计信息"""
        return self._make_request("GET", "/api/landmarks/stats", need_user_id=False)
    
    # ==================== 房源查询接口 ====================
    
    def get_house_by_id(self, house_id: str) -> Dict[str, Any]:
        """根据房源ID获取详情"""
        return self._make_request("GET", f"/api/houses/{house_id}")
    
    def get_house_listings(self, house_id: str) -> Dict[str, Any]:
        """获取房源在各平台的挂牌记录"""
        return self._make_request("GET", f"/api/houses/listings/{house_id}")
    
    def get_houses_by_community(self, community: str, 
                                 listing_platform: Optional[str] = None,
                                 page: int = 1, 
                                 page_size: int = 10) -> Dict[str, Any]:
        """
        按小区名查询可租房源
        
        Args:
            community: 小区名
            listing_platform: 挂牌平台 链家/安居客/58同城
            page: 页码
            page_size: 每页条数
        """
        params = {
            "community": community,
            "page": page,
            "page_size": page_size
        }
        if listing_platform:
            params["listing_platform"] = listing_platform
        return self._make_request("GET", "/api/houses/by_community", params)
    
    def get_houses_by_platform(self, 
                                listing_platform: Optional[str] = None,
                                district: Optional[str] = None,
                                area: Optional[str] = None,
                                min_price: Optional[int] = None,
                                max_price: Optional[int] = None,
                                bedrooms: Optional[str] = None,
                                rental_type: Optional[str] = None,
                                decoration: Optional[str] = None,
                                orientation: Optional[str] = None,
                                elevator: Optional[str] = None,
                                min_area: Optional[int] = None,
                                max_area: Optional[int] = None,
                                property_type: Optional[str] = None,
                                subway_line: Optional[str] = None,
                                max_subway_dist: Optional[int] = None,
                                subway_station: Optional[str] = None,
                                utilities_type: Optional[str] = None,
                                available_from_before: Optional[str] = None,
                                commute_to_xierqi_max: Optional[int] = None,
                                sort_by: Optional[str] = None,
                                sort_order: Optional[str] = None,
                                page: int = 1,
                                page_size: int = 10) -> Dict[str, Any]:
        """
        按条件筛选房源
        
        Args:
            listing_platform: 挂牌平台
            district: 行政区，逗号分隔
            area: 商圈，逗号分隔
            min_price: 最低月租金
            max_price: 最高月租金
            bedrooms: 卧室数，逗号分隔
            rental_type: 整租/合租
            decoration: 装修类型
            orientation: 朝向
            elevator: 是否有电梯 true/false
            min_area: 最小面积
            max_area: 最大面积
            property_type: 物业类型
            subway_line: 地铁线路
            max_subway_dist: 最大地铁距离（米）
            subway_station: 地铁站名
            utilities_type: 水电类型
            available_from_before: 可入住日期上限
            commute_to_xierqi_max: 到西二旗通勤时间上限
            sort_by: 排序字段 price/area/subway
            sort_order: 排序方向 asc/desc
            page: 页码
            page_size: 每页条数
        """
        params = {"page": page, "page_size": page_size}
        
        if listing_platform:
            params["listing_platform"] = listing_platform
        if district:
            params["district"] = district
        if area:
            params["area"] = area
        if min_price is not None:
            params["min_price"] = min_price
        if max_price is not None:
            params["max_price"] = max_price
        if bedrooms:
            params["bedrooms"] = bedrooms
        if rental_type:
            params["rental_type"] = rental_type
        if decoration:
            params["decoration"] = decoration
        if orientation:
            params["orientation"] = orientation
        if elevator:
            params["elevator"] = elevator
        if min_area is not None:
            params["min_area"] = min_area
        if max_area is not None:
            params["max_area"] = max_area
        if property_type:
            params["property_type"] = property_type
        if subway_line:
            params["subway_line"] = subway_line
        if max_subway_dist is not None:
            params["max_subway_dist"] = max_subway_dist
        if subway_station:
            params["subway_station"] = subway_station
        if utilities_type:
            params["utilities_type"] = utilities_type
        if available_from_before:
            params["available_from_before"] = available_from_before
        if commute_to_xierqi_max is not None:
            params["commute_to_xierqi_max"] = commute_to_xierqi_max
        if sort_by:
            params["sort_by"] = sort_by
        if sort_order:
            params["sort_order"] = sort_order
            
        return self._make_request("GET", "/api/houses/by_platform", params)
    
    def get_houses_nearby(self, landmark_id: str,
                          max_distance: Optional[int] = None,
                          listing_platform: Optional[str] = None,
                          page: int = 1,
                          page_size: int = 10) -> Dict[str, Any]:
        """
        以地标为圆心查询附近房源
        
        Args:
            landmark_id: 地标ID或名称
            max_distance: 最大直线距离（米）
            listing_platform: 挂牌平台
            page: 页码
            page_size: 每页条数
        """
        params = {
            "landmark_id": landmark_id,
            "page": page,
            "page_size": page_size
        }
        if max_distance is not None:
            params["max_distance"] = max_distance
        if listing_platform:
            params["listing_platform"] = listing_platform
        return self._make_request("GET", "/api/houses/nearby", params)
    
    def get_nearby_landmarks(self, community: str,
                              landmark_type: Optional[str] = None,
                              max_distance_m: Optional[int] = None) -> Dict[str, Any]:
        """
        查询小区周边地标
        
        Args:
            community: 小区名
            landmark_type: 地标类型 shopping/park
            max_distance_m: 最大距离（米）
        """
        params = {"community": community}
        if landmark_type:
            params["type"] = landmark_type
        if max_distance_m is not None:
            params["max_distance_m"] = max_distance_m
        return self._make_request("GET", "/api/houses/nearby_landmarks", params)
    
    def get_house_stats(self) -> Dict[str, Any]:
        """获取房源统计信息"""
        return self._make_request("GET", "/api/houses/stats")
    
    # ==================== 房源操作接口 ====================
    
    def rent_house(self, house_id: str, listing_platform: str) -> Dict[str, Any]:
        """
        租房操作
        
        Args:
            house_id: 房源ID
            listing_platform: 挂牌平台（必填）
        """
        params = {"listing_platform": listing_platform}
        return self._make_request("POST", f"/api/houses/{house_id}/rent", params)
    
    def terminate_rental(self, house_id: str, listing_platform: str) -> Dict[str, Any]:
        """
        退租操作
        
        Args:
            house_id: 房源ID
            listing_platform: 挂牌平台（必填）
        """
        params = {"listing_platform": listing_platform}
        return self._make_request("POST", f"/api/houses/{house_id}/terminate", params)
    
    def take_offline(self, house_id: str, listing_platform: str) -> Dict[str, Any]:
        """
        下架操作
        
        Args:
            house_id: 房源ID
            listing_platform: 挂牌平台（必填）
        """
        params = {"listing_platform": listing_platform}
        return self._make_request("POST", f"/api/houses/{house_id}/offline", params)


# 定义工具函数的元数据，供LLM使用
TOOLS_DEFINITION = [
    {
        "type": "function",
        "function": {
            "name": "init_houses",
            "description": "重置房源数据到初始状态。在新会话开始时调用，确保数据干净。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_landmarks",
            "description": "获取地标列表，支持按类别和行政区筛选。用于查地铁站、公司、商圈等地标。",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "地标类别：subway(地铁)/company(公司)/landmark(商圈等)"
                    },
                    "district": {
                        "type": "string",
                        "description": "行政区，如 海淀、朝阳"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_landmark_by_name",
            "description": "按名称精确查询地标，如西二旗站、百度。返回地标id、经纬度等，用于后续nearby查房。",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "地标名称，如 西二旗站、国贸"
                    }
                },
                "required": ["name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_landmarks",
            "description": "关键词模糊搜索地标，支持按类别和行政区筛选。",
            "parameters": {
                "type": "object",
                "properties": {
                    "q": {
                        "type": "string",
                        "description": "搜索关键词"
                    },
                    "category": {
                        "type": "string",
                        "description": "地标类别：subway/company/landmark"
                    },
                    "district": {
                        "type": "string",
                        "description": "行政区，如 海淀、朝阳"
                    }
                },
                "required": ["q"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_landmark_by_id",
            "description": "按地标ID查询地标详情。",
            "parameters": {
                "type": "object",
                "properties": {
                    "landmark_id": {
                        "type": "string",
                        "description": "地标ID，如 SS_001、LM_002"
                    }
                },
                "required": ["landmark_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_landmark_stats",
            "description": "获取地标统计信息（总数、按类别分布等）。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_house_by_id",
            "description": "根据房源ID获取单套房源详情。",
            "parameters": {
                "type": "object",
                "properties": {
                    "house_id": {
                        "type": "string",
                        "description": "房源ID，如 HF_2001"
                    }
                },
                "required": ["house_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_house_listings",
            "description": "获取房源在链家/安居客/58同城等各平台的全部挂牌记录。",
            "parameters": {
                "type": "object",
                "properties": {
                    "house_id": {
                        "type": "string",
                        "description": "房源ID，如 HF_2001"
                    }
                },
                "required": ["house_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_houses_by_community",
            "description": "按小区名查询该小区下可租房源。用于指代消解、查某小区地铁信息或隐性属性。",
            "parameters": {
                "type": "object",
                "properties": {
                    "community": {
                        "type": "string",
                        "description": "小区名，如 建清园(南区)、保利锦上(二期)"
                    },
                    "listing_platform": {
                        "type": "string",
                        "description": "挂牌平台：链家/安居客/58同城，不传则默认安居客"
                    },
                    "page": {
                        "type": "integer",
                        "description": "页码，默认1"
                    },
                    "page_size": {
                        "type": "integer",
                        "description": "每页条数，默认10"
                    }
                },
                "required": ["community"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_houses_by_platform",
            "description": "按条件筛选可租房源，支持多维度筛选：行政区、商圈、价格、户型、装修、朝向、电梯、面积、地铁距离、通勤时间等。",
            "parameters": {
                "type": "object",
                "properties": {
                    "listing_platform": {
                        "type": "string",
                        "description": "挂牌平台：链家/安居客/58同城，不传则默认安居客"
                    },
                    "district": {
                        "type": "string",
                        "description": "行政区，逗号分隔，如 海淀,朝阳"
                    },
                    "area": {
                        "type": "string",
                        "description": "商圈，逗号分隔，如 西二旗,上地"
                    },
                    "min_price": {
                        "type": "integer",
                        "description": "最低月租金（元）"
                    },
                    "max_price": {
                        "type": "integer",
                        "description": "最高月租金（元）"
                    },
                    "bedrooms": {
                        "type": "string",
                        "description": "卧室数，逗号分隔，如 1,2"
                    },
                    "rental_type": {
                        "type": "string",
                        "description": "整租或合租"
                    },
                    "decoration": {
                        "type": "string",
                        "description": "装修类型：精装/简装/豪华/毛坯/空房"
                    },
                    "orientation": {
                        "type": "string",
                        "description": "朝向：朝南/朝北/朝东/朝西/南北/东西"
                    },
                    "elevator": {
                        "type": "string",
                        "description": "是否有电梯：true/false"
                    },
                    "min_area": {
                        "type": "integer",
                        "description": "最小面积（平米）"
                    },
                    "max_area": {
                        "type": "integer",
                        "description": "最大面积（平米）"
                    },
                    "property_type": {
                        "type": "string",
                        "description": "物业类型，如 住宅"
                    },
                    "subway_line": {
                        "type": "string",
                        "description": "地铁线路，如 13号线"
                    },
                    "max_subway_dist": {
                        "type": "integer",
                        "description": "最大地铁距离（米），近地铁建议800"
                    },
                    "subway_station": {
                        "type": "string",
                        "description": "地铁站名，如 车公庄站"
                    },
                    "utilities_type": {
                        "type": "string",
                        "description": "水电类型，如 民水民电"
                    },
                    "available_from_before": {
                        "type": "string",
                        "description": "可入住日期上限，YYYY-MM-DD格式"
                    },
                    "commute_to_xierqi_max": {
                        "type": "integer",
                        "description": "到西二旗通勤时间上限（分钟）"
                    },
                    "sort_by": {
                        "type": "string",
                        "description": "排序字段：price/area/subway"
                    },
                    "sort_order": {
                        "type": "string",
                        "description": "排序方向：asc/desc"
                    },
                    "page": {
                        "type": "integer",
                        "description": "页码，默认1"
                    },
                    "page_size": {
                        "type": "integer",
                        "description": "每页条数，默认10"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_houses_nearby",
            "description": "以地标为圆心，查询在指定距离内的可租房源，返回带直线距离、步行距离、步行时间。需先通过地标接口获得landmark_id。",
            "parameters": {
                "type": "object",
                "properties": {
                    "landmark_id": {
                        "type": "string",
                        "description": "地标ID或地标名称"
                    },
                    "max_distance": {
                        "type": "integer",
                        "description": "最大直线距离（米），默认2000"
                    },
                    "listing_platform": {
                        "type": "string",
                        "description": "挂牌平台：链家/安居客/58同城"
                    },
                    "page": {
                        "type": "integer",
                        "description": "页码，默认1"
                    },
                    "page_size": {
                        "type": "integer",
                        "description": "每页条数，默认10"
                    }
                },
                "required": ["landmark_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_nearby_landmarks",
            "description": "查询某小区周边某类地标（商超/公园），按距离排序。用于回答「附近有没有商场/公园」。",
            "parameters": {
                "type": "object",
                "properties": {
                    "community": {
                        "type": "string",
                        "description": "小区名，用于定位基准点"
                    },
                    "landmark_type": {
                        "type": "string",
                        "description": "地标类型：shopping(商超)/park(公园)"
                    },
                    "max_distance_m": {
                        "type": "integer",
                        "description": "最大距离（米），默认3000"
                    }
                },
                "required": ["community"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_house_stats",
            "description": "获取房源统计信息（总套数、按状态/行政区/户型分布、价格区间等）。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "rent_house",
            "description": "租房操作，将房源设为已租状态。必须调用此API才算完成租房，仅对话生成[已租]无效。",
            "parameters": {
                "type": "object",
                "properties": {
                    "house_id": {
                        "type": "string",
                        "description": "房源ID，如 HF_2001"
                    },
                    "listing_platform": {
                        "type": "string",
                        "description": "挂牌平台（必填）：链家/安居客/58同城"
                    }
                },
                "required": ["house_id", "listing_platform"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "terminate_rental",
            "description": "退租操作，将房源恢复为可租状态。",
            "parameters": {
                "type": "object",
                "properties": {
                    "house_id": {
                        "type": "string",
                        "description": "房源ID，如 HF_2001"
                    },
                    "listing_platform": {
                        "type": "string",
                        "description": "挂牌平台（必填）：链家/安居客/58同城"
                    }
                },
                "required": ["house_id", "listing_platform"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "take_offline",
            "description": "下架操作，将房源设为下架状态。",
            "parameters": {
                "type": "object",
                "properties": {
                    "house_id": {
                        "type": "string",
                        "description": "房源ID，如 HF_2001"
                    },
                    "listing_platform": {
                        "type": "string",
                        "description": "挂牌平台（必填）：链家/安居客/58同城"
                    }
                },
                "required": ["house_id", "listing_platform"]
            }
        }
    }
]
