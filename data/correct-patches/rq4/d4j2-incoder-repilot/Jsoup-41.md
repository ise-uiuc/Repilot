# Repilot Patch

```
        return Objects.equals(tag, element.tag);
```

# Developer Patch

```
        return tag.equals(element.tag);
```

# Context

```
--- bug/Jsoup-41/src/main/java/org/jsoup/nodes/Element.java

+++ fix/Jsoup-41/src/main/java/org/jsoup/nodes/Element.java

@@ -1172,7 +1172,7 @@

 
         Element element = (Element) o;
 
-        return this == o;
+        return Objects.equals(tag, element.tag);
     }
 
     @Override
```

# Note

